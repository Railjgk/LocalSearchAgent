#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run Beijing/Qingdao Gaode POI extraction with the existing mock-data contract.

The heavy lifting stays in `build_supply_from_gaode_v2.py`: search execution,
resume, dedupe, enrichment, and WeekendFlow mock file generation. This wrapper
only prepares city overlays so Beijing and Qingdao can use the same output
schema as the Shanghai seed while adding city-specific categories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "gaode_supply_extraction_config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "mock_data"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiments" / "artifacts" / "gaode_city_poi"


CITY_PROFILES: dict[str, dict[str, Any]] = {
    "北京": {
        "slug": "beijing",
        "areas": [
            {"id": "beijing_guomao", "name": "国贸", "district": "朝阳", "location": "116.461100,39.909200", "tags": ["核心商圈", "餐饮密集", "约会"]},
            {"id": "beijing_sanlitun", "name": "三里屯", "district": "朝阳", "location": "116.453700,39.933600", "tags": ["核心商圈", "朋友聚会", "咖啡"]},
            {"id": "beijing_wudaokou", "name": "五道口", "district": "海淀", "location": "116.337700,39.992900", "tags": ["学生", "朋友聚会", "餐饮密集"]},
            {"id": "beijing_zhongguancun", "name": "中关村", "district": "海淀", "location": "116.316700,39.983600", "tags": ["核心商圈", "学生", "快餐"]},
            {"id": "beijing_wangjing", "name": "望京", "district": "朝阳", "location": "116.480200,39.996800", "tags": ["社区生活圈", "家庭", "日料"]},
            {"id": "beijing_xidan", "name": "西单", "district": "西城", "location": "116.374100,39.914800", "tags": ["核心商圈", "餐饮密集", "亲子"]},
            {"id": "beijing_wangfujing", "name": "王府井", "district": "东城", "location": "116.412300,39.914800", "tags": ["核心商圈", "文化街区", "餐饮密集"]},
            {"id": "beijing_qianmen", "name": "前门", "district": "东城", "location": "116.397500,39.900500", "tags": ["citywalk", "文化街区", "本地文化"]},
            {"id": "beijing_shichahai", "name": "什刹海", "district": "西城", "location": "116.386200,39.941000", "tags": ["citywalk", "咖啡", "小众"]},
            {"id": "beijing_chaoyang_park", "name": "朝阳公园", "district": "朝阳", "location": "116.484000,39.933300", "tags": ["亲子", "公园", "户外"]},
        ],
        "area_grids": [
            {
                "id": "beijing_dense_grid",
                "name": "北京中心城区网格",
                "district": "北京",
                "bounds": {"lng_min": 116.10, "lng_max": 116.70, "lat_min": 39.72, "lat_max": 40.15},
                "step_km": 3.2,
                "max_centers": 280,
                "tags": ["核心商圈", "餐饮密集", "朋友聚会", "轻食", "社区生活圈", "家庭", "亲子", "日料", "约会", "citywalk", "咖啡", "学生", "快餐", "炸鸡", "公园", "小众", "文化街区", "户外", "本地文化"],
            }
        ],
    },
    "青岛": {
        "slug": "qingdao",
        "areas": [
            {"id": "qingdao_may_fourth_square", "name": "五四广场", "district": "市南", "location": "120.382600,36.067100", "tags": ["核心商圈", "约会", "景观"]},
            {"id": "qingdao_mixc", "name": "万象城", "district": "市南", "location": "120.379400,36.068600", "tags": ["核心商圈", "餐饮密集", "轻食"]},
            {"id": "qingdao_taidong", "name": "台东", "district": "市北", "location": "120.351300,36.087200", "tags": ["餐饮密集", "学生", "夜市"]},
            {"id": "qingdao_badaguan", "name": "八大关", "district": "市南", "location": "120.350000,36.052100", "tags": ["citywalk", "文化街区", "小众"]},
            {"id": "qingdao_old_town", "name": "中山路老城", "district": "市南", "location": "120.319300,36.066200", "tags": ["citywalk", "本地文化", "咖啡"]},
            {"id": "qingdao_licun", "name": "李村", "district": "李沧", "location": "120.421200,36.161000", "tags": ["社区生活圈", "家庭", "餐饮密集"]},
            {"id": "qingdao_laoshan", "name": "崂山商圈", "district": "崂山", "location": "120.467400,36.107900", "tags": ["家庭", "亲子", "户外"]},
            {"id": "qingdao_huangdao", "name": "黄岛金沙滩", "district": "黄岛", "location": "120.197700,35.960600", "tags": ["亲子", "户外", "景观"]},
        ],
        "area_grids": [
            {
                "id": "qingdao_dense_grid",
                "name": "青岛中心城区网格",
                "district": "青岛",
                "bounds": {"lng_min": 120.10, "lng_max": 120.55, "lat_min": 35.85, "lat_max": 36.25},
                "step_km": 3.0,
                "max_centers": 220,
                "tags": ["核心商圈", "餐饮密集", "朋友聚会", "轻食", "社区生活圈", "家庭", "亲子", "日料", "约会", "citywalk", "咖啡", "学生", "快餐", "炸鸡", "公园", "小众", "文化街区", "户外", "夜市", "本地文化", "景观"],
            }
        ],
    },
}


def category(
    category_id: str,
    expected_type: str,
    label: str,
    types: str,
    text_keywords: list[str],
    around_keywords: list[str],
    area_tags: list[str],
    quota_weight: float = 1.0,
) -> dict[str, Any]:
    return {
        "id": category_id,
        "expected_type": expected_type,
        "category": label,
        "types": types,
        "quota_weight": quota_weight,
        "text_keywords": text_keywords,
        "around_keywords": around_keywords,
        "area_tags": area_tags,
    }


SPECIAL_CATEGORIES: dict[str, list[dict[str, Any]]] = {
    "北京": [
        category("restaurant_beijing_duck", "restaurant", "北京烤鸭", "050000", ["北京 烤鸭", "北京 京味烤鸭", "北京 全聚德 烤鸭", "北京 大董 烤鸭", "北京 四季民福 烤鸭"], ["烤鸭", "京味烤鸭"], ["本地文化", "核心商圈", "餐饮密集", "家庭"], 1.25),
        category("restaurant_copper_hotpot", "restaurant", "涮羊肉铜锅", "050000", ["北京 涮羊肉", "北京 铜锅涮肉", "北京 老北京涮肉", "北京 羊蝎子", "北京 羊肉串"], ["涮羊肉", "铜锅涮肉", "老北京涮肉", "羊蝎子", "羊肉串"], ["本地文化", "餐饮密集", "朋友聚会", "家庭"], 1.15),
        category("restaurant_beijing_snack", "restaurant", "京味小吃", "050000", ["北京 京味小吃", "北京 炸酱面", "北京 卤煮", "北京 门钉肉饼", "北京 驴打滚"], ["京味小吃", "炸酱面", "卤煮", "门钉肉饼"], ["本地文化", "学生", "快餐", "社区生活圈"], 0.9),
        category("activity_hutong_citywalk", "activity", "胡同城市漫步", "060000|080000|110000|140000", ["北京 胡同 citywalk", "北京 胡同 咖啡", "北京 前门 citywalk", "北京 什刹海 citywalk", "北京 南锣鼓巷"], ["胡同", "citywalk", "南锣鼓巷", "前门", "什刹海"], ["citywalk", "文化街区", "本地文化", "咖啡", "小众"], 1.05),
        category("activity_beijing_performance", "activity", "相声演出剧场", "080000|140000", ["北京 相声", "北京 脱口秀", "北京 剧场", "北京 演出", "北京 亲子剧场"], ["相声", "脱口秀", "剧场", "演出", "亲子剧场"], ["本地文化", "约会", "朋友聚会", "亲子"], 0.9),
    ],
    "青岛": [
        category("restaurant_qingdao_seafood", "restaurant", "青岛海鲜", "050000", ["青岛 海鲜", "青岛 海鲜大排档", "青岛 海鲜家常菜", "青岛 本地海鲜", "青岛 蒸汽海鲜"], ["海鲜", "海鲜大排档", "本地海鲜", "蒸汽海鲜"], ["本地文化", "景观", "餐饮密集", "家庭"], 1.3),
        category("restaurant_qingdao_beer_house", "restaurant", "啤酒屋", "050000", ["青岛 啤酒屋", "青岛 啤酒街", "青岛 精酿啤酒", "青岛 海鲜啤酒屋"], ["啤酒屋", "啤酒街", "精酿啤酒", "海鲜啤酒屋"], ["本地文化", "夜市", "朋友聚会", "餐饮密集"], 1.1),
        category("restaurant_qingdao_local", "restaurant", "鲁菜饺子鲅鱼水饺", "050000", ["青岛 鲁菜", "青岛 鲅鱼水饺", "青岛 饺子", "青岛 家常菜", "青岛 蛤蜊"], ["鲁菜", "鲅鱼水饺", "饺子", "家常菜", "蛤蜊"], ["本地文化", "家庭", "社区生活圈", "餐饮密集"], 1.05),
        category("activity_seaside_citywalk", "activity", "海边城市漫步", "060000|080000|110000|140000", ["青岛 海边 citywalk", "青岛 八大关 citywalk", "青岛 栈桥 citywalk", "青岛 小麦岛 citywalk", "青岛 老城漫步"], ["海边", "八大关", "栈桥", "小麦岛", "老城漫步", "citywalk"], ["citywalk", "景观", "文化街区", "小众", "户外"], 1.05),
        category("activity_qingdao_beer_culture", "activity", "啤酒文化夜市", "080000|110000|140000", ["青岛 啤酒博物馆", "青岛 啤酒节", "青岛 夜市", "青岛 台东夜市", "青岛 啤酒文化"], ["啤酒博物馆", "啤酒节", "夜市", "台东夜市", "啤酒文化"], ["本地文化", "夜市", "朋友聚会", "景观"], 0.9),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run city Gaode extraction for WeekendFlow POI mock supply.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cities", nargs="+", default=["青岛", "北京"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--max-calls-per-city", type=int, default=800)
    parser.add_argument("--sleep-sec", type=float, default=0.34)
    parser.add_argument("--request-timeout-sec", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--base-only", action="store_true", help="Use only generic city categories, without special city categories.")
    parser.add_argument("--special-only", action="store_true", help="Run only city-specific category overlays.")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--skip-rebuild-from-raw", action="store_true")
    return parser.parse_args()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML is required to read the extraction config.")
    if not path.exists():
        raise SystemExit(f"Extraction config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"Extraction config must be an object: {path}")
    return config


def replace_city_token(value: Any, city: str) -> Any:
    if isinstance(value, str):
        return value.replace("上海", city)
    if isinstance(value, list):
        return [replace_city_token(item, city) for item in value]
    if isinstance(value, dict):
        return {key: replace_city_token(item, city) for key, item in value.items()}
    return value


def selected_categories(base_categories: list[dict[str, Any]], city: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str] | None]:
    special = SPECIAL_CATEGORIES.get(city, [])
    if args.base_only:
        return base_categories, None
    if args.special_only:
        return special, [item["id"] for item in special]
    return base_categories + special, None


def city_config(base_config: dict[str, Any], city: str, args: argparse.Namespace) -> dict[str, Any]:
    profile = CITY_PROFILES.get(city)
    if not profile:
        supported = ", ".join(sorted(CITY_PROFILES))
        raise SystemExit(f"Unsupported city {city!r}. Supported cities: {supported}")

    config = replace_city_token(base_config, city)
    config["name"] = f"weekendflow_{profile['slug']}_supply_v1"
    defaults = dict(config.get("defaults") or {})
    defaults["city"] = city
    defaults["include_area_grids"] = True
    config["defaults"] = defaults
    config["areas"] = profile["areas"]
    config["area_grids"] = profile["area_grids"]

    categories, include_category_ids = selected_categories(list(config.get("categories") or []), city, args)
    config["categories"] = categories

    category_count = max(1, len(categories if include_category_ids is None else include_category_ids))
    mode = {
        "max_calls": args.max_calls_per_city,
        "max_pages_per_query": 1,
        "per_category_call_cap": max(1, args.max_calls_per_city // category_count),
        "include_area_grids": True,
    }
    if include_category_ids is not None:
        mode["include_category_ids"] = include_category_ids

    modes = dict(config.get("modes") or {})
    modes["city_poi"] = mode
    config["modes"] = modes
    return config


def output_dir_for(output_root: Path, city: str) -> Path:
    return (output_root / f"gaode_supply_{CITY_PROFILES[city]['slug']}_v1").resolve()


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


def run_build(config_path: Path, city: str, output_dir: Path, args: argparse.Namespace, extra: list[str] | None = None) -> int:
    cmd = [
        sys.executable,
        "experiments/build_supply_from_gaode_v2.py",
        "--config",
        str(config_path),
        "--mode",
        "city_poi",
        "--city",
        city,
        "--output-dir",
        str(output_dir),
        "--max-calls",
        str(args.max_calls_per_city),
        "--sleep-sec",
        str(args.sleep_sec),
        "--request-timeout-sec",
        str(args.request_timeout_sec),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-sec",
        str(args.retry_backoff_sec),
    ]
    if args.dry_run_plan:
        cmd.append("--dry-run-plan")
    if extra:
        cmd.extend(extra)
    return run(cmd)


def run_city(args: argparse.Namespace, city: str, temp_root: Path) -> dict[str, Any]:
    base_config = load_config(args.config)
    config = city_config(base_config, city, args)
    config_path = temp_root / f"gaode_supply_{CITY_PROFILES[city]['slug']}.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    output_dir = output_dir_for(args.output_root, city)
    rebuild_code = 0
    if not args.dry_run_plan and not args.skip_rebuild_from_raw and (output_dir / "raw_pois.jsonl").exists():
        rebuild_code = run_build(config_path, city, output_dir, args, ["--rebuild-from-raw"])
        if rebuild_code != 0:
            return city_result(city, output_dir, rebuild_code=rebuild_code, api_code=1, validation_code=0)

    api_code = run_build(config_path, city, output_dir, args)
    validation_code = 0
    if api_code == 0 and not args.dry_run_plan and not args.skip_validation:
        validation_code = run(
            [
                sys.executable,
                "experiments/validate_gaode_supply_run.py",
                "--data-dir",
                str(output_dir),
                "--skip-eval",
            ]
        )
    return city_result(city, output_dir, rebuild_code=rebuild_code, api_code=api_code, validation_code=validation_code)


def city_result(city: str, output_dir: Path, *, rebuild_code: int, api_code: int, validation_code: int) -> dict[str, Any]:
    report = read_json(output_dir / "build_report.json", {})
    return {
        "city": city,
        "output_dir": str(output_dir.relative_to(REPO_ROOT) if output_dir.is_relative_to(REPO_ROOT) else output_dir),
        "exit_codes": {
            "rebuild_from_raw": rebuild_code,
            "api_run": api_code,
            "validation": validation_code,
        },
        "activity_count": report.get("activity_count"),
        "restaurant_count": report.get("restaurant_count"),
        "deduped_records": report.get("deduped_records"),
        "attempted_calls": report.get("attempted_calls"),
        "successful_calls": report.get("successful_calls"),
        "error_calls": report.get("error_calls"),
    }


def main() -> int:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="gaode-city-poi-") as temp_dir_raw:
        temp_root = Path(temp_dir_raw)
        for city in args.cities:
            results.append(run_city(args, city, temp_root))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": started_at,
        "max_calls_per_city": args.max_calls_per_city,
        "dry_run_plan": args.dry_run_plan,
        "base_only": args.base_only,
        "special_only": args.special_only,
        "results": results,
    }
    summary_path = args.summary_dir / f"gaode_city_poi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote city POI summary: {summary_path}")

    failed = [
        item
        for item in results
        if any(code != 0 for code in item["exit_codes"].values())
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
