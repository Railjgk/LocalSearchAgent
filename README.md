
# 美团AI黑客松 - 本地生活Agent (C模块：执行层)

## 项目简介

本项目是美团AI黑客松赛道六「现在就出发：AI本地路线智能规划」的执行层模块（C模块）。

**核心职责**：将上游规划好的方案，通过 Mock API 模拟真实交易链路，完成「预订 → 执行 → 分享」的完整闭环。

**技术亮点**：
- ✅ 完整的 LangGraph 12节点编排
- ✅ 带库存状态的内存数据库（同一时段重复预订会失败）
- ✅ 满位自动排队机制
- ✅ 支持家庭/朋友双场景
- ✅ 完整的执行日志追踪

---

## 架构概览

C模块为**执行层**，整体执行流程如下：

1. 接收来自 B 模块的 `selected_plan` 规划方案
2. 经 **Tool Router** 生成 `action_sequence` 动作序列
3. 通过 **Mock API Layer** 调用模拟接口
4. **Execution Manager** 汇总执行结果、做状态判断
5. **Share Generator** 生成最终分享文案 `final_share_message`


---

### 文件结构
```txt
src/
├── state.py                    # 全局状态定义（12模块共享）
├── graph.py                    # LangGraph 12节点编排
├── nodes/
│   ├── tool_router.py          # 工具路由：生成动作序列
│   ├── mock_api_layer.py       # Mock API 调用层
│   ├── execution_manager.py    # 执行管理器：结果汇总
└── tools/
    └── mock_apis.py            # Mock API 实现（库存、预订、排队）

run.py                          # 运行入口
requirements.txt                # 依赖列表
```
---

## 核心功能

### 1. Mock API 层 (`mock_apis.py`)

提供以下模拟接口：

| API | 功能 | 状态模拟 |
|-----|------|----------|
| `search_activities` | 搜索活动 | 支持 kid_friendly, low_intensity 筛选 |
| `search_restaurants` | 搜索餐厅 | 支持 low_calorie, family_friendly 筛选 |
| `reserve_restaurant` | 预订餐厅 | 库存扣减、满位自动排队 |
| `order_activity_ticket` | 购买门票 | 库存扣减、场次售罄检测 |
| `order_addon_service` | 附加服务 | 蛋糕、鲜花等 |
| `call_taxi` | 叫车 | 成功率 90% |
| `check_availability` | 查询库存 | 返回剩余库存和排队长度 |
| `check_route` | 路线检查 | 返回距离和时间 |
| `take_queue_number` | 取号排队 | 返回排队号和预估等待时间 |
| `cancel_order` | 取消订单 | 释放库存 |

**关键特性**：
- 内存数据库模拟真实库存（同一时段第二次预订会失败）
- 满位时自动取号排队
- 支持取消订单并释放库存

### 2. Tool Router (`tool_router.py`)

- 读取 `selected_plan` 中的 `timeline`
- 根据活动类型生成 `action_sequence`
- 自动解析时间格式（`14:00-16:00` → `14:00`）
- 家庭场景自动添加蛋糕订单

### 3. Execution Manager (`execution_manager.py`)

- 遍历执行 `action_sequence`
- 调用对应的 Mock API
- 统计成功/失败数量
- 判断执行状态：`success` / `partial` / `failed`

### 4. Share Generator (`share_generator.py`)

- 根据执行状态生成不同文案
- 家庭场景温馨风格、朋友场景活泼风格
- 支持成功/部分失败/全失败三种模板

---

## 安装与运行

### 环境要求

- Python 3.10+
- pip

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-repo/meituan-hackathon.git
cd meituan-hackathon

# 2. 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
python run.py
```

### 预期输出

```
============================================================
🚀 美团AI黑客松 - 本地生活Agent启动
============================================================

🔧 [8] Tool Router: 解析方案，生成执行动作列表...
📡 [9] Mock API Layer: 调用API...
⚙️ [9-10] Execution Manager: 执行动作序列...
📱 [11] Share Generator: 生成分享消息...

============================================================
📊 执行结果
============================================================

📝 执行日志:
   ✅ Tool Router: 生成2个执行动作
   ▶ 执行: reserve_restaurant - 轻食日料餐厅
      ✅ reserve_restaurant 成功: 已预订轻食日料餐厅 17:30，3位
   ▶ 执行: order_activity_ticket - 亲子陶艺体验馆
      ✅ order_activity_ticket 成功: 已购亲子陶艺体验馆 14:00场次，3张票
   📊 执行统计: 成功2/2, 失败0
   📊 执行状态: success

💬 分享消息: 🎉 搞定了！下午安排好了：14:00 亲子陶艺体验馆 → 17:30 轻食日料餐厅。已经帮你订好了～

============================================================
✅ 执行完成
============================================================
```

---

## 与 A/B 模块的数据契约

### 输入格式（来自 B 模块）

```python
selected_plan = {
    "timeline": [
        {
            "type": "play",           # 活动类型：play / eat
            "poi_id": "act_001",      # POI ID
            "activity": "活动名称",    # 显示用
            "time": "14:00"           # 精确时间（匹配 POI 的 time_slots）
        }
    ]
}

constraints = {
    "people_count": 3,                # 人数
    "child_age": 5,                   # 孩子年龄（家庭场景）
    "mom_diet": "low_calorie"         # 饮食约束（家庭场景）
}
```

### 输出格式

```python
{
    "execution_status": "success",    # success / partial / failed
    "final_share_message": "分享文案",
    "tool_results": {...},            # 各工具执行结果详情
    "execution_log": [...]            # 执行日志
}
```

---

## 依赖列表 (`requirements.txt`)

```txt
langgraph>=0.0.20
langchain>=0.1.0
openai>=1.0.0
python-dotenv>=1.0.0
```

---

## 后续优化方向

| 优先级 | 优化项 | 说明 |
|--------|--------|------|
| P1 | 接入真实地图API | 替代 `check_route` 的随机距离 |
| P1 | 增加用户确认节点 | 预订前让用户确认 |
| P2 | 接入 LongCat 生成文案 | 让分享消息更自然 |
| P2 | 订单状态流转 | pending → confirmed → completed |
| P3 | 并发库存锁 | 模拟高并发场景 |

---


