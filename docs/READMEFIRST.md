# LocalSearchAgent 项目模块分析

## 1. 项目整体功能
LocalSearchAgent 是基于 LangGraph 改造的本地化 AI 智能体开发框架，用于构建有状态、可持久化、长时间运行的智能体系统。
核心能力：
- 支持多智能体协同、流程编排、工具调用
- 提供断点恢复、人工介入、记忆持久化
- 纯本地私有化部署，不上云，保障数据隐私
- 提供 CLI 命令行一键启动、构建、部署

## 2. 目录结构说明
# 项目目录结构

LocalSearchAgent-main/
├── .github/           GitHub 自动化、Issue/PR 模板
├── libs/              核心功能库
│   ├── checkpoint/           状态持久化基础接口
│   ├── checkpoint-sqlite/    SQLite 状态存储
│   ├── checkpoint-postgres/  PostgreSQL 状态存储
│   ├── cli/                  命令行交互工具
│   ├── langgraph/            智能体图执行引擎
│   ├── prebuilt/             开箱即用智能体 API
│   ├── sdk-py/               Python 调用 SDK
│   └── sdk-js/               JavaScript 调用 SDK

## 3. 核心模块作用
### libs/langgraph（核心引擎）
整个系统的调度中心，负责：
- 智能体流程编排（StateGraph）
- 多节点状态流转、消息传递
- 条件分支、循环、多智能体协同

### libs/cli（命令行入口）
提供命令行工具，支持：
- 本地开发调试
- 一键构建 Docker 镜像
- 快速部署运行

### libs/checkpoint-*（状态持久化）
负责：
- 智能体状态、记忆存储
- 程序崩溃后断点恢复
- 支持 SQLite / PostgreSQL 两种存储

## 4. 代码执行流程
1. 用户通过 CLI 命令启动（langgraph up/dev/build）
2. 加载配置文件 langgraph.json
3. 初始化图结构 StateGraph
4. 执行智能体节点，按流程自动调度
5. 状态实时保存到数据库
6. 支持暂停、人工干预、断点恢复

## 5. 技术栈总结
- 开发语言：Python + JavaScript/TypeScript
- 架构模式：图编排引擎（Pregel 模型）
- 数据存储：SQLite、PostgreSQL
- 部署方式：CLI + Docker
- 核心特点：本地私有化、持久化、状态可恢复、隐私安全