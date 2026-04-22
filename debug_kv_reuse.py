"""
KV Cache reuse 精度诊断脚本
=========================
用法：把此脚本中各"探针"作为 print 语句或 assert 嵌入到对应位置，
或者直接 import 后在 notebook / 调试器 中调用。

每个探针独立可用，按照从"最外层"到"最内层"的顺序逐步缩小问题范围。
"""

import torch


# ===========================================================================
# 探针 0：验证 L_pos 与 DiT 正常路径的 cached_prompt_len 是否一致
# ===========================================================================
# 位置：pipeline_hunyuan_image3.py → _forward_with_kv_reuse，
#       在 total_seq_len 计算之后（约第 1379 行）插入。
#
# 思路：
#   正常路径（无 KV 复用）的 cached_prompt_len = total_seq_len - num_image_tokens - 1
#                                              = L_text_dit + NUM_SPECIAL_TOKENS
#   其中 L_text_dit 是 DiT 自己 tokenize 同一 prompt 后的文本长度。
#   如果 L_pos（来自 AR）≠ L_text_dit，两条路径的位置序号就会错位。
#
# 如何判读：
#   "L_pos from AR" 与 "L_text_dit from DiT tokenize" 应完全相等。
#   如果不等，请确认：
#     - AR 是否包含了 system prompt；DiT 是否也包含了相同的 system prompt
#     - AR 是否把 CoT 生成 token 也放进了 KV cache（它们不应该被送进来）
#     - AR tokenize 时是否加了 <bos>；DiT 是否同样加了 <bos>

def probe_0_check_L_pos_vs_dit_text_len(
    pipeline,          # HunyuanImage3Text2ImagePipeline instance
    prompt: str,
    system_prompt: str | None,
    cot_text: str | None,
    image_size: tuple,
    L_pos: int,        # 从 AR KV cache 读到的 key_cache[i].shape[0]
    NUM_SPECIAL_TOKENS: int = 3,
):
    """在 DiT 侧重新 tokenize，与 L_pos 对比。"""
    # DiT 正常路径 tokenize
    image_info = pipeline.image_processor.build_image_info(image_size)
    out = pipeline._tkwrapper.apply_chat_template(
        batch_prompt=[prompt],
        batch_message_list=None,
        mode="gen_image",
        batch_gen_image_info=[image_info],
        batch_cond_image_info=None,
        batch_system_prompt=[system_prompt] if system_prompt else None,
        batch_cot_text=[cot_text] if cot_text else None,
        max_length=None,
        bot_task="auto",
        image_base_size=pipeline.config.image_base_size,
        sequence_template="pretrain",
        cfg_factor=1,
        drop_think=False,
    )
    tokens = out["output"].tokens  # [1, total_seq_len]
    total_seq_len_dit = tokens.shape[1]
    num_image_tokens = image_info.token_height * image_info.token_width + (
        1 if image_info.add_timestep_token else 0
    )
    # cached_prompt_len in normal path = total_seq_len - num_image_tokens - 1 (eoi)
    cached_prompt_len_dit = total_seq_len_dit - num_image_tokens - 1
    L_text_dit = cached_prompt_len_dit - NUM_SPECIAL_TOKENS

    print(f"[Probe-0] L_pos(AR KV length)   = {L_pos}")
    print(f"[Probe-0] L_text_dit(DiT normal) = {L_text_dit}")
    print(f"[Probe-0] cached_prompt_len(DiT) = {cached_prompt_len_dit}  "
          f"(= L_text_dit + NUM_SPECIAL_TOKENS = {L_text_dit + NUM_SPECIAL_TOKENS})")
    print(f"[Probe-0] total_seq_len(DiT)     = {total_seq_len_dit}")
    print(f"[Probe-0] num_image_tokens       = {num_image_tokens}")
    if L_pos == L_text_dit:
        print("[Probe-0] ✅ L_pos == L_text_dit: AR KV 长度与 DiT 文本长度匹配")
    else:
        diff = L_pos - L_text_dit
        print(f"[Probe-0] ❌ MISMATCH: L_pos - L_text_dit = {diff}")
        print("         可能原因：")
        print("           1. AR KV cache 包含了 CoT / recaption token")
        print("           2. system_prompt 在 AR/DiT 两侧不一致")
        print("           3. AR tokenize 时 BOS 处理与 DiT 不同")
    return L_text_dit, cached_prompt_len_dit


# ===========================================================================
# 探针 1：打印注入后第 0 层 KV cache 的统计信息
# ===========================================================================
# 位置：pipeline_hunyuan_image3.py → _forward_with_kv_reuse，
#       inject_prompt_kv_cache 循环结束后（约第 1423 行之后）插入。
#
# 思路：
#   检查 AR 传来的 KV 是否含有合理的数值（非全零、非 NaN/Inf）。
#   如果 pos_key_cache[0] 全零或 NaN，说明 KV transfer 本身出了问题。

def probe_1_check_injected_kv_stats(
    pos_key_cache: list,   # past_kv.key_cache
    pos_value_cache: list, # past_kv.value_cache
    neg_key_cache: list | None = None,
    layer_idx: int = 0,
):
    """打印第 layer_idx 层注入 KV 的基本统计量。"""
    # 找第一个非 None 的层
    pk = pos_key_cache[layer_idx]
    if pk is None:
        for pk in pos_key_cache:
            if pk is not None:
                break
    pv = pos_value_cache[layer_idx]
    if pv is None:
        for pv in pos_value_cache:
            if pv is not None:
                break

    print(f"\n[Probe-1] Layer {layer_idx} pos_key shape  : {tuple(pk.shape)}")
    print(f"[Probe-1] Layer {layer_idx} pos_key dtype  : {pk.dtype}")
    print(f"[Probe-1] Layer {layer_idx} pos_key has NaN: {torch.isnan(pk.float()).any().item()}")
    print(f"[Probe-1] Layer {layer_idx} pos_key has Inf: {torch.isinf(pk.float()).any().item()}")
    print(f"[Probe-1] Layer {layer_idx} pos_key abs max : {pk.float().abs().max().item():.6f}")
    print(f"[Probe-1] Layer {layer_idx} pos_key abs mean: {pk.float().abs().mean().item():.6f}")
    print(f"[Probe-1] Layer {layer_idx} pos_key all zero: {(pk == 0).all().item()}")

    if neg_key_cache is not None:
        nk = neg_key_cache[layer_idx]
        if nk is None:
            for nk in neg_key_cache:
                if nk is not None:
                    break
        print(f"[Probe-1] Layer {layer_idx} neg_key shape  : {tuple(nk.shape)}")
        print(f"[Probe-1] Layer {layer_idx} neg_key abs max : {nk.float().abs().max().item():.6f}")
        print(f"[Probe-1] Layer {layer_idx} neg_key all zero: {(nk == 0).all().item()}")


# ===========================================================================
# 探针 2：对比 attention_mask 的形状和屏蔽分布
# ===========================================================================
# 位置：pipeline_hunyuan_image3.py → _forward_with_kv_reuse，
#       attention_mask 构建完毕后（约第 1477 行之后）插入。
#
# 思路：
#   打印每个 batch slot 的 True（可见）位置比例，以及被 mask 的列索引。
#   与正常路径相比，mask 结构应完全一致（masked 列：special tokens + eoi）。

def probe_2_check_attention_mask(
    attention_mask: torch.Tensor,  # [bsz, 1, num_image_tokens, total_seq_len]
    L_text: int,
    NUM_SPECIAL_TOKENS: int,
    num_image_tokens: int,
    total_seq_len: int,
):
    """打印 attention_mask 的 masked 列分布。"""
    bsz = attention_mask.shape[0]
    print(f"\n[Probe-2] attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"[Probe-2] total_seq_len={total_seq_len}, L_text={L_text}, "
          f"NUM_SPECIAL={NUM_SPECIAL_TOKENS}, num_image_tokens={num_image_tokens}")

    for b in range(bsz):
        mask_b = attention_mask[b, 0, 0, :]   # [total_seq_len]  (all query rows same)
        false_cols = (mask_b == False).nonzero(as_tuple=True)[0].tolist()
        true_count = mask_b.sum().item()
        print(f"[Probe-2] batch[{b}] True(visible)={true_count}/{total_seq_len}, "
              f"masked cols={false_cols}")

    # 期望：batch[0] masked = [L_text..L_text+NUM_SPECIAL, -1(eoi)]
    #       batch[1] (CFG) 额外 masked = [L_neg..L_text]
    expected_always_masked = list(range(L_text, L_text + NUM_SPECIAL_TOKENS)) + [total_seq_len - 1]
    print(f"[Probe-2] Expected masked cols (pos branch): {expected_always_masked}")


# ===========================================================================
# 探针 3：对比正常路径 vs KV 复用路径的第一步 attention 输出（逐层）
# ===========================================================================
# 位置：hunyuan_image3_transformer.py → ImageKVCacheManager.__call__，
#       attn_output 计算之后（约第 1165 行之后）插入。
#
# 思路：
#   用一个全局 dict 分别在两次推理中记录每层第一步的 attn_output。
#   推理结束后对比两者的 L2 距离。如果某层开始出现大误差，那层就是问题所在。
#
# 使用方式：
#   在推理前 import 此脚本并初始化：
#     from debug_kv_reuse import attn_record, probe_3_record_attn_output, probe_3_compare_runs
#     attn_record.clear()
#   两次推理结束后调用：
#     probe_3_compare_runs()

attn_record: dict[str, dict[int, torch.Tensor]] = {}  # run_tag -> {layer_idx: tensor}
_current_run_tag: str = "normal"
_record_step: int = 0  # 只记录 step 0（第一步）

def set_run_tag(tag: str):
    """在推理开始前调用，tag='normal' 或 'kv_reuse'。"""
    global _current_run_tag, _record_step
    _current_run_tag = tag
    _record_step = 0
    attn_record.setdefault(tag, {})

def probe_3_record_attn_output(
    layer_idx: int,
    step_idx: int,         # 当前 denoising step（传入即可，从 0 开始）
    attn_output: torch.Tensor,  # [bs*q_len, heads, head_dim]
):
    """在 ImageKVCacheManager.__call__ 末尾调用。"""
    if step_idx == 0:
        attn_record[_current_run_tag][layer_idx] = attn_output.detach().float().cpu()

def probe_3_compare_runs(top_k: int = 5):
    """两次推理后调用，打印各层误差。"""
    if "normal" not in attn_record or "kv_reuse" not in attn_record:
        print("[Probe-3] 需先分别以 tag='normal' 和 'kv_reuse' 跑一次推理")
        return
    normal = attn_record["normal"]
    reuse = attn_record["kv_reuse"]
    layers = sorted(set(normal.keys()) & set(reuse.keys()))
    errors = []
    for li in layers:
        a, b = normal[li], reuse[li]
        if a.shape != b.shape:
            print(f"[Probe-3] Layer {li}: shape mismatch {a.shape} vs {b.shape}")
            continue
        l2 = (a - b).norm().item()
        rel = l2 / (a.norm().item() + 1e-8)
        errors.append((rel, li, l2))
    errors.sort(reverse=True)
    print(f"\n[Probe-3] Top-{top_k} layers with largest relative error (step 0 attn_output):")
    print(f"{'Layer':>6}  {'RelErr':>10}  {'AbsL2':>12}")
    for rel, li, l2 in errors[:top_k]:
        print(f"{li:>6}  {rel:>10.4f}  {l2:>12.4f}")
    if errors:
        worst_rel = errors[0][0]
        if worst_rel < 0.01:
            print("[Probe-3] ✅ 所有层误差 < 1%，attention 输出基本一致")
        elif worst_rel < 0.05:
            print("[Probe-3] ⚠️  最大误差 1%–5%，可能有轻微数值差异")
        else:
            print("[Probe-3] ❌ 误差 > 5%，attention 存在显著分歧")


# ===========================================================================
# 探针 4：在 _update_image_kv_caches 中打印 cached_key 布局
# ===========================================================================
# 位置：hunyuan_image3_transformer.py → _update_image_kv_caches，
#       函数开头插入。
#
# 思路：
#   检查注入的 cached_key（来自 inject_prompt_kv_cache）的形状是否符合预期。
#   非零段应在 [0, L_text)，[L_text + NUM_SPECIAL_TOKENS, ...) 之后才是 image。
#   [L_text, L_text + NUM_SPECIAL_TOKENS) 应全零（special token 零填充）。

def probe_4_check_cached_key_layout(
    cached_key: torch.Tensor,   # image_kv_cache_map[0]，来自 _update_image_kv_caches
    bs: int,
    L_text: int,
    NUM_SPECIAL_TOKENS: int = 3,
):
    """在 _update_image_kv_caches 的第一次调用时（step 1）执行。"""
    total_cached = cached_key.shape[0]
    cached_prompt_len = total_cached // bs - 1   # eoi 占 1
    print(f"\n[Probe-4] cached_key.shape={tuple(cached_key.shape)}, bs={bs}, "
          f"cached_prompt_len={cached_prompt_len}")

    # 取 pos 分支（前 cached_prompt_len+1 个 token）
    pk = cached_key[:cached_prompt_len + 1].float()   # [cached_prompt_len + 1, kv_heads, head_dim]

    # 文本段 [0, L_text)
    text_part = pk[:L_text]
    special_part = pk[L_text: L_text + NUM_SPECIAL_TOKENS]
    eoi_part = pk[L_text + NUM_SPECIAL_TOKENS: L_text + NUM_SPECIAL_TOKENS + 1]

    print(f"[Probe-4] text_part   [{0}:{L_text}]  "
          f"abs_max={text_part.abs().max().item():.6f}  "
          f"all_zero={text_part.abs().max().item() == 0}")
    print(f"[Probe-4] special_part[{L_text}:{L_text+NUM_SPECIAL_TOKENS}]  "
          f"abs_max={special_part.abs().max().item():.6f}  "
          f"all_zero={special_part.abs().max().item() == 0}")
    print(f"[Probe-4] eoi_part                  "
          f"abs_max={eoi_part.abs().max().item():.6f}  "
          f"all_zero={eoi_part.abs().max().item() == 0}")

    if special_part.abs().max().item() > 1e-6:
        print("[Probe-4] ❌ special_part 不为零！inject_prompt_kv_cache 的零填充可能出问题")
    else:
        print("[Probe-4] ✅ special_part 全零，符合预期")

    if text_part.abs().max().item() < 1e-6:
        print("[Probe-4] ❌ text_part 全零！AR KV cache 传输可能失败（全零缓存）")
    else:
        print("[Probe-4] ✅ text_part 有效（非全零）")


# ===========================================================================
# 探针 5：对比两条路径 position_ids 和 RoPE cos/sin
# ===========================================================================
# 位置：pipeline_hunyuan_image3.py → _forward_with_kv_reuse，
#       position_ids 和 custom_pos_emb 构建完毕后插入。
#
# 思路：
#   正常路径的 position_ids = [0, 1, ..., total_seq_len-1]（全序列）。
#   KV 复用路径的 position_ids = [L_text+NUM_SPECIAL, ..., L_text+NUM_SPECIAL+num_image_tokens-1]。
#   这两者的图像段应该完全重合。
#   cos/sin[L_text+NUM_SPECIAL : L_text+NUM_SPECIAL+num_image_tokens] 在两条路径应相等。

def probe_5_check_position_ids_and_rope(
    position_ids_reuse: torch.Tensor,   # [bsz, num_image_tokens]
    cos_reuse: torch.Tensor,            # [total_seq_len, head_dim/2] or similar
    sin_reuse: torch.Tensor,
    cos_normal: torch.Tensor | None = None,  # 来自正常路径（可选）
    sin_normal: torch.Tensor | None = None,
    L_text: int = 0,
    NUM_SPECIAL_TOKENS: int = 3,
    num_image_tokens: int = 0,
):
    img_start = L_text + NUM_SPECIAL_TOKENS
    img_end = img_start + num_image_tokens
    print(f"\n[Probe-5] position_ids_reuse[0, :5] = {position_ids_reuse[0, :5].tolist()}")
    print(f"[Probe-5] position_ids_reuse[0, -3:] = {position_ids_reuse[0, -3:].tolist()}")
    print(f"[Probe-5] expected range: [{img_start}, {img_end})")

    if position_ids_reuse[0, 0].item() != img_start:
        print(f"[Probe-5] ❌ position_ids 起始 {position_ids_reuse[0,0].item()} ≠ 期望 {img_start}")
    else:
        print(f"[Probe-5] ✅ position_ids 起始正确")

    if cos_normal is not None:
        # 取图像段对比
        cos_r = cos_reuse[:, img_start:img_end] if cos_reuse.dim() >= 2 else cos_reuse[img_start:img_end]
        cos_n = cos_normal[:, img_start:img_end] if cos_normal.dim() >= 2 else cos_normal[img_start:img_end]
        err = (cos_r.float() - cos_n.float()).abs().max().item()
        print(f"[Probe-5] cos image-seg max diff = {err:.6e}")
        if err < 1e-5:
            print("[Probe-5] ✅ RoPE cos 图像段完全一致")
        else:
            print("[Probe-5] ❌ RoPE cos 图像段存在差异")


# ===========================================================================
# 探针 6：单步 noise level 检查（对比 timestep / sigma）
# ===========================================================================
# 位置：pipeline_hunyuan_image3.py，在 __call__ / _generate 的 denoising 循环内。
#
# 思路：
#   KV 复用路径的 timestep schedule 由 `pipeline` 的 scheduler 驱动，
#   正常路径也是同一个 scheduler。检查两者 timestep 序列是否完全相同。

def probe_6_check_timestep_schedule(
    timesteps_reuse: list | torch.Tensor,
    timesteps_normal: list | torch.Tensor | None = None,
):
    t_r = list(timesteps_reuse) if not isinstance(timesteps_reuse, list) else timesteps_reuse
    print(f"\n[Probe-6] KV-reuse timesteps (first 5): {t_r[:5]}")
    print(f"[Probe-6] KV-reuse timesteps (last 3) : {t_r[-3:]}")
    if timesteps_normal is not None:
        t_n = list(timesteps_normal) if not isinstance(timesteps_normal, list) else timesteps_normal
        if t_r == t_n or all(abs(a - b) < 1e-4 for a, b in zip(t_r, t_n)):
            print("[Probe-6] ✅ timestep schedule 完全一致")
        else:
            print(f"[Probe-6] ❌ timestep schedule 不一致！normal={t_n[:5]}...")


# ===========================================================================
# 如何使用
# ===========================================================================
#
# 建议按以下顺序逐步执行：
#
# Step A — 确认 L_pos 是否匹配（最常见根因）
#   在 _forward_with_kv_reuse 中，于 total_seq_len 计算后加入：
#
#     from debug_kv_reuse import probe_0_check_L_pos_vs_dit_text_len
#     probe_0_check_L_pos_vs_dit_text_len(
#         pipeline=self,
#         prompt=req.prompts[0].get("prompt", "") if req.prompts else "",
#         system_prompt=system_prompt,   # 从 forward() 透传过来
#         cot_text=req.prompts[0].get("extra", {}).get("ar_generated_text") if req.prompts else None,
#         image_size=(height, width),
#         L_pos=L_pos,
#     )
#
# Step B — 确认 AR KV 传输内容正常
#   在 inject 循环后加入：
#
#     from debug_kv_reuse import probe_1_check_injected_kv_stats
#     probe_1_check_injected_kv_stats(pos_key_cache, pos_value_cache,
#                                     neg_key_cache if use_cfg else None)
#
# Step C — 确认 attention_mask 被正确屏蔽
#   在 attention_mask 构建后加入：
#
#     from debug_kv_reuse import probe_2_check_attention_mask
#     probe_2_check_attention_mask(attention_mask, L_text, NUM_SPECIAL_TOKENS,
#                                  num_image_tokens, total_seq_len)
#
# Step D — 逐层对比两条路径的 attention 输出（需要两次推理）
#   先运行正常路径：
#     from debug_kv_reuse import set_run_tag; set_run_tag("normal")
#   然后在 ImageKVCacheManager.__call__ 末尾加入：
#     from debug_kv_reuse import probe_3_record_attn_output
#     probe_3_record_attn_output(layer_idx=<当前层索引>, step_idx=<当前step>, attn_output=attn_output)
#   再运行 KV 复用路径：
#     set_run_tag("kv_reuse")
#   两次推理后：
#     from debug_kv_reuse import probe_3_compare_runs; probe_3_compare_runs()
#
# Step E — 检查 cached_key 布局（第 2 个 step 调用时触发）
#   在 _update_image_kv_caches 开头加：
#     from debug_kv_reuse import probe_4_check_cached_key_layout
#     probe_4_check_cached_key_layout(cached_key, bs, L_text=<L_text从外部传入>)
#
# 返回结果时，请提供：
#   1. Probe-0 的输出（L_pos vs L_text_dit）
#   2. Probe-1 的输出（AR KV 数值统计）
#   3. Probe-2 的输出（attention_mask masked cols）
#   4. Probe-3 的输出（逐层误差排行，如果可以跑两次推理）
#   5. Probe-4 的输出（cached_key 布局校验）

if __name__ == "__main__":
    print("此脚本是诊断工具库，请按照文件末尾的'如何使用'说明将各探针嵌入到对应位置后执行。")
