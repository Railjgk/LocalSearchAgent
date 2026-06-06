# LocalSearchBench Evaluation Toolkit

Evaluation toolkit for [LocalSearchBench](https://github.com/localsearchbench/localsearchbench).

## 📝 Abstract

Recent advances in large reasoning models (LRMs) have enabled agentic search systems to perform complex multi-step reasoning across multiple sources. However, most research focuses on general information retrieval and rarely explores vertical domains with unique challenges. In this work, we focus on local life services and introduce LocalSearchBench, which encompasses diverse and complex business scenarios. Real-world queries in this domain are often ambiguous and require multi-hop reasoning across merchants and products, remaining challenging and not fully addressed.

This toolkit provides scripts for evaluating agents on the LocalSearchBench benchmark.

## 🧱 Data Fields

| Field                 | Type   | Description                                                       |
| --------------------- | ------ | ----------------------------------------------------------------- |
| Hop Count             | int64  | Number of chained search hops used to solve the query (3–5).      |
| Difficulty            | string | L3 (standard multi-hop) or L4 (complex constraints).              |
| City                  | string | One of the nine supported Chinese cities.                         |
| Question              | string | Natural-language request describing the user's goal.              |
| Multi-hop search path | string | Serialized trace of each hop's query + retrieved evidence.        |
| Answer                | string | Final recommendation sequence with addresses, hours, and pricing. |

## 🚀 Usage

Please refer to the individual Python scripts for usage details.

## 📂 Project Structure

- `config/`: Configuration files
- `*.py`: Evaluation scripts

## 🏅 Leaderboard Snapshot

| Model             | Avg tool calls | Avg rounds | Correctness | Completeness | Faithfulness |
| ----------------- | -------------- | ---------- | ----------- | ------------ | ------------ |
| DeepSeek-V3.1     | 3.43           | 4.02       | **34.34**   | 80.00        | 60.80        |
| Qwen-Plus-Latest  | 2.59           | 3.12       | 32.79       | **80.94**    | 68.68        |
| LongCat-Large-32K | 2.73           | 3.22       | 33.19       | 80.51        | 60.80        |
| GPT-4.1           | 1.72           | 2.70       | 26.76       | 75.42        | 72.63        |
| Gemini-2.5-Pro    | 1.89           | 2.86       | 26.09       | 77.93        | **78.26**    |

## 📜 License

MIT License.
