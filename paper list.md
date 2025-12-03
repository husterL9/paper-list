# ASPLOS

## 2025

<span id ="asplos2501">[vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](#asplos2501-1)</span>

<span id ="asplos2502">[Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management](#asplos2502-1)</span>

<span id ="asplos2503">[M5: Mastering Page Migration and Memory  Management for CXL-based Tiered Memory Systems](#asplos2503-1)</span>

## [<span id ="asplos2501-1">vAttention: Dynamic Memory Management for  Serving LLMs without PagedAttention</span>](#asplos2501)

        本文提出针对 LLM 服务的动态内存管理方案 vAttention，旨在解决 PagedAttention 需重写内核、开销高、可移植性差的问题。其核心是通过 CUDA VMM API 解耦虚拟与物理内存分配，提前预留连续虚拟内存缓冲区以保留 KV 缓存虚拟连续性，同时按需映射物理内存缓解碎片；并通过内存分配与计算重叠、延迟回收、小页面支持等优化，降低运行时延迟。实验表明，在 Yi-6B 等模型上，vAttention 预填充阶段比 FlashAttention-2/FlashInfer 的 Paged 版本快 1.17-1.36 倍，端到端离线吞吐量提升 1.13-1.23 倍，还能无缝支持 FlashAttention-3 等新内核，兼顾性能、简洁性与可移植性。

### 一、研究背景与现有方案痛点

LLM 服务中，KV 缓存占 GPU 推理内存的主要部分，其动态增长特性（单请求逐 token 扩展、总长度未知）导致内存分配难题：

1. **静态分配**（如 Orca、FasterTransformer）按模型最大上下文长度预留内存，引发严重内部碎片，限制批处理大小和吞吐量；
2. **PagedAttention 方案**（vLLM 提出，被 TensorRT-LLM 等广泛采用）通过按需分配小内存块缓解碎片，但存在根本缺陷：
   - 需重写注意力内核以支持非连续 KV 缓存访问，难以跟进最新优化（如 vLLM 的 Paged 内核比 FlashAttention-2 慢 2.8 倍）；
   - 服务框架需额外实现内存管理，重复 OS 的虚实地址转换功能，增加冗余；
   - 运行时开销显著：GPU 侧因块表查询、寄存器溢出等慢 37%（FlashAttention-2 预填充阶段）至 42%（FlashInfer 预填充阶段），CPU 侧因块表准备等增加 10%-30% 延迟；
   - 可移植性差，新内核（如 FlashAttention-3）发布时无 PagedAttention 支持。

### 二、vAttention 核心设计

核心思路：**保留 KV 缓存的虚拟内存连续性，同时动态分配物理内存**，通过 CUDA 虚拟内存管理（VMM）API 解耦虚实内存分配，避免 PagedAttention 的非连续布局缺陷。

1. **内存分配策略**
   
   - 虚拟内存：提前预留超大连续缓冲区（按最大批处理大小和模型最大上下文长度配置），利用 64 位系统充足的虚拟地址空间（单进程可达 128TB），无需担心虚拟内存碎片；
   - 物理内存：运行时按需分配，仅在请求需要时将物理页面映射到虚拟缓冲区，避免提前占用。

2. **关键技术细节**
   
   - 支持多粒度页面：修改开源 CUDA 统一内存驱动，新增 64KB/128KB/256KB 小页面支持（默认 CUDA VMM 仅支持 2MB 大页面），降低物理内存碎片；
   - 请求级 KV 缓存索引：通过唯一 reqId 定位批处理中每个请求的 KV 缓存子张量，确保地址访问连续性；
   - 兼容现有框架：作为 Python 库集成到 vLLM 等服务框架，提供 init/alloc_reqid/free_reqid/step 等简洁 API，无需修改模型或注意力内核。

3. **针对性优化**
   
   - 隐藏分配延迟：利用解码阶段内存需求的可预测性，通过后台线程将内存分配与计算重叠；预填充阶段采用延迟回收 + 预分配策略，复用已释放的物理页面；
   - 缓解碎片：小页面支持使分配粒度匹配 KV 缓存增长特性（单 token 仅需数十 KB），且无 TLB 抖动风险；
   - 支持连续批处理：借助 FlashAttention 的 cache_batch_idx API，处理请求退出后的虚拟内存 "空洞"，保持批处理灵活性。

### 三、实验结果

基于 Yi-6B/Llama-3-8B/Yi-34B 模型，在 A100（单卡 / 双卡 NVLink）和 H100 GPU 上的测试表明：

1. **性能优势**
   
   - 预填充阶段：长上下文（192K）下，vAttention 比 FlashAttention-2 的 Paged 版本快 1.24-1.26 倍，比 FlashInfer 的 Paged 版本快 1.17-1.36 倍；
   - 解码阶段：与 FlashAttention-2 的 Paged 版本性能相当，比 vLLM 的 Paged 内核快 1.53-1.99 倍，比 FlashInfer 的 Paged 版本快 1.23 倍；
   - 端到端吞吐量：离线长上下文任务（arXiv 摘要）中，比 FlashAttention-2 Paged 快 1.13-1.18 倍，比 FlashInfer Paged 快 1.14-1.23 倍；在线场景下中位数延迟降低 28%-42%。

2. **可移植性**
   
   - 无需修改代码即可支持 FlashAttention-3（Hopper 架构优化内核），在 H100 上比 FlashAttention-2 Paged 版本吞吐量提升 1.26-1.5 倍。

3. **资源效率**
   
   - 小页面（64KB）使最大批处理大小提升 1.18-1.28 倍；内存分配带宽达 7.6GB/s，远超 LLM 推理需求（750MB/s）。

### 四、核心贡献

1. 提出虚实内存解耦的 KV 缓存管理方案，兼顾连续性（无内核修改）和动态性（无碎片）；
2. 解决 CUDA VMM 的延迟和大页面碎片问题，提供 LLM 专用优化；
3. 实现简单、可移植、高性能的替代方案，支持现有主流注意力内核（FlashAttention-2/3、FlashInfer），降低 LLM 服务的部署和维护成本。
   
    

## [<span id ="asplos2502-1">Accelerating LLM Serving for Multi-turn Dialogues  with Efficient Resource Management</span>](#asplos2502)

        本文针对现有 LLM 服务框架在处理多轮对话时存在的历史注意力键值对（KVs）重计算开销大、FCFS 调度导致 GPU 内存利用率低（头阻塞）两大问题，提出了名为 FlashGen 的解决方案：通过设计包含 GPU、CPU 内存和 SSD 的多级 KV 缓存（FlashGen-Cache），动态选择缓存恢复与重计算以减少冗余计算，同时采用请求重排序调度（FlashGen-Sched），在优先调度可运行短请求提升内存利用率的同时，通过抢占机制避免长请求饥饿；基于 Azure 实例（双 A100 GPU 等配置），在 OPT、Llama-2 系列模型及 ShareGPT 等数据集上的实验表明，FlashGen 在相似延迟下，对 OPT 30B 和 Llama-2 70B 的吞吐量分别提升 1.63 倍和 2.85 倍，显著优化了多轮对话场景下的 LLM 服务性能

### 一、研究背景与核心问题

1. **多轮对话的 LLM 服务挑战**：随着 LLM 在聊天机器人等场景的广泛应用，长上下文（如多轮对话）处理需求激增，但现有框架（如 vLLM、TensorRT-LLM）存在两大关键效率问题：
   
   - **KV 重计算开销**：多轮对话中，用户查询会包含历史对话内容，导致提示词长度 “放大”，现有框架因 GPU 内存有限无法缓存所有历史注意力键值对（KVs），需重复计算，耗费大量资源。
   - **GPU 内存利用率低**：采用先到先服务（FCFS）调度策略时，长提示词请求会阻塞后续短请求，导致 GPU 内存闲置（头阻塞问题），尤其在高负载下缓存竞争加剧，利用率进一步下降。

2. **数据支撑**：基于 ShareGPT 真实对话数据集的分析显示，对话会话平均包含 7 轮、中位数 3 轮，多轮对话的提示词长度较单轮增长 99 倍，历史对话内容占总输入 tokens 的一半以上，验证了 “提示词放大” 问题的严重性。

### 二、核心解决方案：FlashGen

FlashGen 通过**多级 KV 缓存管理**和**请求重排序调度**两大核心技术，高效利用 GPU、CPU（DRAM）和 SSD 资源，解决上述问题。

#### 1. 多级 KV 缓存（FlashGen-Cache）

- **设计目标**：避免历史 KVs 重复计算，通过多级存储分层缓存，平衡内存成本与访问 latency。
- **缓存层级**：
  - 一级缓存（GPU 内存）：缓存当前运行请求的 KVs 及已完成请求的可回收 KVs，优先命中以减少传输开销。
  - 二级缓存（CPU 内存）：异步复制 GPU 生成的 KVs，GPU 内存不足时可快速恢复，通过流水线技术重叠 KV 传输与模型计算，隐藏延迟。
  - 三级缓存（SSD）：当 CPU 内存不足以存储所有历史 KVs 时，异步归档不常用 KVs；通过 CPU 内存 “预加载” 机制，避免直接从 SSD 读取的高延迟，必要时动态选择 “重计算” 而非 “SSD 读取” 以优化性能。
- **关键优化**：批量感知 KV 恢复、主动缓存策略（生成时即复制到 CPU），减少内存回收与传输开销。

#### 2. 请求重排序调度（FlashGen-Sched）

- **设计目标**：解决头阻塞问题，提升 GPU 内存利用率，同时保证请求公平性。
- **核心策略**：
  - 贪心重排序：当队列头部的长请求因内存不足无法执行时，优先调度后续可放入空闲内存的短请求（“提升请求”），避免内存闲置。
  - 无饥饿机制：实时跟踪 GPU 内存使用，当空闲内存 + 提升请求占用内存足以容纳被阻塞的长请求时，抢占提升请求，优先执行长请求，避免其饥饿。
- **效果**：GPU 内存利用率从 vLLM 的 88% 提升至 98% 以上，批量请求规模平均增加 1.06~1.15 倍。

### 三、实验验证与结果

1. **实验环境**：Azure 实例（2×A100 GPU、440GB CPU 内存、2×960GB NVMe SSD），模型包括 OPT（13B/30B/66B/175B）、Llama-2（13B/70B），数据集涵盖 ShareGPT（多轮对话）、Alpaca、HumanEval。

2. **核心性能指标**：
   
   - 吞吐量：在相似延迟下，FlashGen 对 OPT 30B 和 Llama-2 70B 的吞吐量分别提升 1.63 倍和 2.85 倍。
   - 延迟：P95 首 token 延迟（TTFT）较 vLLM 降低 77%（OPT 30B）和 66%（Llama-2 13B）；单 token 生成延迟（TPOT）的 P99 值从 vLLM 的 608ms 降至 103ms（OPT 30B）。
   - 缓存效果：GPU+CPU+SSD 三级缓存的 KV 命中率显著高于单一层级，高负载下仍能维持稳定命中，减少重计算比例。

3. **对比基准**：优于 vLLM（基线）和 CachedAttention（同类 KV 缓存方案），尤其在高负载、长上下文场景下，因动态调度与多级缓存的协同优化，性能优势更明显。

### 四、相关工作与结论

1. **相关工作对比**：
   
   - KV 复用：CachedAttention 仅支持 CPU/SSD 缓存，未动态选择重计算；SGLang 仅依赖 GPU 缓存，不支持异构存储。
   - 调度优化：现有迭代级调度未解决多轮对话的头阻塞问题；Sarathi-Serve 聚焦长提示词拆分，不涉及请求重排序。
   - 内存优化：PagedAttention 优化内存分配，但未解决历史 KV 缓存与调度协同问题。

2. **研究结论**：
   
   - FlashGen 通过多级 KV 缓存和请求重排序的协同设计，有效解决了多轮对话中 “KV 重计算” 和 “GPU 内存闲置” 两大核心问题。
   - 在长上下文、高负载场景下性能优势显著，为 LLM 多轮服务（如聊天机器人）提供了高效、低成本的解决方案，随着对话轮数增加，优化价值更突出。
   
   

## [<span id ="asplos2503-1">M5: Mastering Page Migration and Memory  Management for CXL-based Tiered Memory Systems</span>](#asplos2503)

​	本文针对 CXL 基于分层内存系统中传统 CPU 驱动页面迁移方案精度低、开销大且无法区分稀疏页面的问题，首先提出基于 FPGA 的 CXL 驱动页面与字访问计数方案（PAC 与 WAC），以精准统计 CXL DRAM 中 4KB 页面和 64B 字的访问次数；接着通过 PAC 与 WAC 揭示了 ANB、DAMON 等 CPU 驱动方案误判温页面、盲目迁移稀疏页面及性能开销显著的缺陷；最后设计并实现 M5 平台，该平台依托 CXL 控制器中的硬件热页面 / 热字跟踪器（HPT/HWT）及软件 M5-manager，能低成本、高精度识别热页面与区分页面稀疏性，实验表明 M5 平均比最优 CPU 驱动方案（DAMON）多识别 47% 热页面且提升 14% 性能，为 CXL 分层内存系统的高效管理提供实用解决方案

### 一、研究背景与挑战

1. **技术背景**：数据中心应用对 DRAM 容量和带宽需求持续增长，但传统 DDR 接口已接近缩放极限。CXL 作为基于 PCIe 的新型内存接口，能以更少引脚提供与 DDR 相当的带宽，可低成本扩展内存容量，但 CXL DRAM 访问延迟比 DDR 高 2-3 倍，形成 “DDR（快内存）+ CXL DRAM（慢内存）” 的分层内存系统。
2. **核心挑战**：需高效的页面迁移方案，将频繁访问的 “热页面” 从 CXL DRAM 迁移到 DDR，以降低性能损失。但现有 CPU 驱动的页面迁移方案存在三大问题：
   - 易将 “温页面” 误判为 “热页面”，识别精度低；
   - 无法区分 “稀疏页面”（仅少量 64B 字频繁访问）和 “密集页面”，迁移稀疏页面会造成缓存污染和内存浪费；
   - 识别热页面的过程消耗大量 CPU 周期，性能开销显著，可能抵消迁移收益。

### 二、核心贡献

#### 1. CXL 驱动的页面与字访问计数方案（PAC 与 WAC）

- 基于 FPGA 的 CXL 设备实现，利用 CXL 控制器的近内存处理能力，精准、透明地统计 CXL DRAM 中每个 4KB 页面（PAC）和 64B 字（WAC）的访问次数。
- 相比动态二进制插桩、采样等传统方法，PAC 和 WAC 无需干扰应用执行，计数精度更高，为评估页面迁移方案提供了黄金标准。

#### 2. 揭示 CPU 驱动页面迁移方案的缺陷

通过 PAC 和 WAC 的实测分析：

- **识别精度低**：代表性方案 ANB（自动 NUMA 平衡）和 DAMON 识别的 “热页面”，其实际访问量仅为 PAC 判定的 Top-K 热页面的 21% 和 29%，本质是误判温页面；
- **稀疏页面迁移问题**：Redis 等应用中 86% 的页面仅 25% 以下的字被访问，但 CPU 驱动方案无法区分，盲目迁移导致缓存污染；
- **性能开销大**：ANB 和 DAMON 分别使内核 CPU 周期增加 159% 和 277%，导致 Redis 等延迟敏感应用的 p99 延迟上升 34%-39%，部分应用执行时间延长超 8%。

#### 3. M5 平台设计与实现

M5 是支持 CXL 驱动页面迁移方案开发的硬件 - 软件协同平台，核心目标是解决 CPU 驱动方案的缺陷，包含两大组件：

- **硬件组件**：热页面跟踪器（HPT）和热字跟踪器（HWT）
  - 基于 Count-Min Sketch（CM-Sketch）算法，低成本跟踪 Top-K 热页面和热字，避免 CPU 驱动方案的高开销；
  - 运行于 CXL 控制器，无需修改 CPU 架构，支持 400MHz 以上速率，满足内存访问实时性要求。
- **软件组件**：M5-manager
  - 包含 Monitor（监控内存带宽密度等指标）、Nominator（结合 HPT/HWT 识别密集热页面）、Elector（动态调整迁移频率和策略）、Promoter（与 Linux 内核交互执行页面迁移）；
  - 提供灵活接口，支持用户自定义迁移策略，并给出 4 条核心优化准则（如根据带宽密度决定迁移优先级、区分稀疏 / 密集页面迁移等）。

### 三、实验验证与结果

1. **实验环境**：基于 Intel 第 4 代 Xeon 处理器的双路服务器，搭配 Intel Agilex-7 FPGA（CXL 设备），测试 12 个内存密集型基准测试（含 SPEC CPU 2017、Redis、图计算等）。
2. **关键结果**：
   - **识别精度**：M5 的 CM-Sketch-based HPT 识别热页面的访问量占比达 0.72，比 ANB/DAMON（平均 0.49）高 47%；
   - **性能提升**：M5 平均比 DAMON（现有最优 CPU 驱动方案）提升 14% 性能，比 ANB 提升 20%；对 Redis 等延迟敏感应用，性能提升达 43%；
   - **开销优势**：M5 识别热页面的 CPU 开销可忽略，避免了 ANB/DAMON 的内核资源占用问题。

### 四、研究结论

M5 通过 CXL 控制器的硬件辅助跟踪与灵活的软件策略，解决了传统 CPU 驱动页面迁移方案的精度低、开销大、无法区分稀疏页面等问题，为 CXL 分层内存系统提供了高效、实用的内存管理解决方案，且兼容性强，可与现有 Linux 内核功能（如 MGLRU）协同工作，具备工业应用潜力。