# Qwen3-Omni API Server - 快速参考

## 📋 新增文件

1. **qwen3_omni_api_server.py** - 主程序（集成所有功能）
   - API Server 模式：提供 HTTP API 接口
   - Benchmark 模式：独立运行性能测试
   - 并发控制、QPS 限流、性能统计

2. **client.py** - API 客户端
   - 连接到 API Server
   - 发送 TTS 请求
   - 启动和获取 benchmark 结果

3. **USAGE.py** - 使用说明（运行查看）
4. **run_benchmark.sh** - Shell 脚本快速启动

---

## 🚀 快速开始

### 方式 1：启动 API Server（推荐）

```bash
# 启动服务器（默认 0.0.0.0:8000）
python qwen3_omni_api_server.py --mode server

# 或指定端口
python qwen3_omni_api_server.py --mode server --port 8080
```

然后在另一个终端中使用客户端：

```bash
# 单个 TTS 请求
python client.py --prompt "Hello world" --speaker Ethan

# 启动 benchmark
python client.py --start-benchmark --prompts-file top100.txt

# 获取结果
python client.py --get-results
```

### 方式 2：独立 Benchmark 模式

```bash
# 运行完整 benchmark（默认：1, 4, 8 并发 × 0.1, 0.2, 0.3, 0.4 QPS）
python qwen3_omni_api_server.py --mode benchmark --prompts_file top100.txt

# 自定义配置
python qwen3_omni_api_server.py --mode benchmark \
  --prompts_file top100.txt \
  --num_prompts 50 \
  --concurrency_levels 1 4 8 \
  --qps_levels 0.1 0.2 0.3 0.4
```

### 方式 3：使用 Shell 脚本

```bash
# 启动服务器
bash run_benchmark.sh server

# 运行 benchmark
bash run_benchmark.sh benchmark top100.txt 100

# 自定义配置
bash run_benchmark.sh benchmark top100.txt 50 "1 4 8" "0.1 0.2 0.3 0.4"
```

---

## 📊 支持的配置

| 配置 | 值 |
|------|-----|
| **并发级别** | 1, 4, 8 |
| **QPS 速率** | 0.1, 0.2, 0.3, 0.4 |
| **总组合数** | 3 × 4 = 12 种配置 |

---

## 🎯 API 端点（Server 模式）

### 健康检查
```bash
curl -X POST http://localhost:8000/health
```

### 单个 TTS 请求
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello world",
    "speaker": "Ethan"
  }'
```

### 启动 Benchmark
```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "prompts_file": "top100.txt",
    "concurrency_levels": [1, 4, 8],
    "qps_levels": [0.1, 0.2, 0.3, 0.4],
    "output_dir": "benchmark_results"
  }'
```

### 获取 Benchmark 结果
```bash
curl http://localhost:8000/benchmark/results
```

---

## 📈 输出格式

Benchmark 结果保存为 JSON：

```json
{
  "concurrency_1_qps_0.1": {
    "concurrency": 1,
    "qps": 0.1,
    "total_requests": 100,
    "successful_requests": 100,
    "failed_requests": 0,
    "min_latency": 0.5,
    "max_latency": 3.2,
    "mean_latency": 1.5,
    "median_latency": 1.4,
    "p95_latency": 2.1,
    "p99_latency": 3.0,
    "throughput": 0.95,
    "total_duration": 105.2,
    "success_rate": 1.0,
    "avg_thinker_tokens": 128,
    "avg_talker_tokens": 256,
    "avg_code2wav_tokens": 512
  },
  ...
}
```

---

## 💡 性能指标说明

| 指标 | 单位 | 说明 |
|------|------|------|
| **latency** | 秒 | 请求处理时间 |
| **throughput** | req/s | 每秒成功请求数 |
| **success_rate** | % | 成功率（目标 >95%）|
| **tokens** | 个 | 各模块生成的 token 数量 |

---

## 🔧 命令行参数

### Server 模式
```
--mode server              运行为 API 服务器
--host HOST               服务器地址（默认：0.0.0.0）
--port PORT               服务器端口（默认：8000）
--model_path PATH         模型路径
```

### Benchmark 模式
```
--mode benchmark           运行为独立 benchmark
--prompts_file PATH       提示词文件
--num_prompts N           处理的提示词数量
--concurrency_levels L... 并发级别列表
--qps_levels Q...         QPS 级别列表
--output_dir PATH         输出目录
--model_path PATH         模型路径
```

---

## 📝 典型工作流

### 完整 Benchmark 流程

```bash
# 1. 准备提示词文件
# top100.txt 中每行一个提示词

# 2. 运行 benchmark
python qwen3_omni_api_server.py --mode benchmark \
  --prompts_file top100.txt \
  --num_prompts 100 \
  --output_dir results_$(date +%Y%m%d_%H%M%S)

# 3. 查看结果
cat results_*/benchmark_results.json | python -m json.tool
```

### API Server 流程

```bash
# Terminal 1: 启动服务器
python qwen3_omni_api_server.py --mode server

# Terminal 2: 发送请求
python client.py --health-check
python client.py --prompt "test"
python client.py --start-benchmark --prompts-file top100.txt

# Terminal 3: 查看结果
python client.py --get-results
```

---

## 🎯 特点

✅ **单文件实现** - 所有代码在 qwen3_omni_api_server.py 中  
✅ **两种运行模式** - API Server 或独立 Benchmark  
✅ **完整的并发控制** - QPS 限流 + 并发管理  
✅ **详细的性能指标** - 延迟、吞吐量、成功率等  
✅ **易于集成** - 标准 FastAPI 接口  
✅ **自动结果保存** - JSON 格式自动保存  

---

## 🐛 常见问题

**Q: 如何快速测试？**
A: 运行最少的配置：
```bash
python qwen3_omni_api_server.py --mode benchmark \
  --prompts_file top100.txt \
  --num_prompts 5 \
  --concurrency_levels 1 \
  --qps_levels 0.1
```

**Q: 如何只测试某个配置？**
A: 指定单个并发和 QPS：
```bash
python qwen3_omni_api_server.py --mode benchmark \
  --prompts_file top100.txt \
  --concurrency_levels 4 \
  --qps_levels 0.2
```

**Q: 结果在哪里？**
A: 在 `benchmark_results.json` 中，位于 `--output_dir` 指定的目录。

**Q: 如何增加更多的并发级别？**
A: 直接在命令行中添加：
```bash
--concurrency_levels 1 2 4 8 16
```

---

## 📚 查看更多

运行以下命令查看详细使用说明：
```bash
python USAGE.py
```

---

**最后提示：** 所有功能都在一个文件中，无需配置文件，开箱即用！🚀
