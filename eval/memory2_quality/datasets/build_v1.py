"""Generate the reviewed synthetic v1 matrix as JSONL."""
from __future__ import annotations

import json
from pathlib import Path

NOW = "2026-07-01T10:00:00+08:00"


def case(case_id, category, message, expected, *, initial=None, source="manual", query=None):
    payload = {
        "case_id": case_id,
        "category": category,
        "description": message,
        "source": source,
        "reference_time": NOW,
        "initial_memories": initial or [],
        "sessions": [{"session_id": "s1", "timestamp": NOW, "messages": [{"role": "user", "content": message}], "consolidate_after": True}],
        "expected_write": expected,
        "recall_probes": [],
        "tags": [category],
    }
    required = expected.get("required", [])
    if required:
        payload["recall_probes"] = [{"probe_id": "recall", "query": query or message, "top_k": 5, "required_memory_labels": [item["label"] for item in required]}]
    elif expected.get("forbidden"):
        payload["recall_probes"] = [{"probe_id": "negative_recall", "query": query or message, "top_k": 5, "forbidden_facts": expected["forbidden"][0]["facts"]}]
    return payload


def required(label, kind, fact):
    return {"required": [{"label": label, "memory_type": kind, "facts": [fact]}], "allowed_new_count": {"min": 1, "max": 2}}


def build():
    rows = []
    typed = {
        "event": [("昨天完成了硕士论文答辩", "用户完成了硕士论文答辩"), ("上周参观了桂林博物馆", "用户参观了桂林博物馆"), ("今天完成了项目部署", "用户完成了项目部署"), ("六月参加了一次技术面试", "用户六月参加了技术面试")],
        "profile": [("我是计算机专业的硕士研究生", "用户是计算机专业硕士研究生"), ("我现在长期住在桂林", "用户长期住在桂林"), ("我的主力手机是安卓手机", "用户使用安卓手机"), ("我目前正在准备秋招", "用户目前准备秋招")],
        "preference": [("我通常喜欢安静的餐厅", "用户喜欢安静的餐厅"), ("我一直不喜欢过甜的饮料", "用户不喜欢过甜的饮料"), ("看技术文章时我更喜欢中文", "用户偏好中文技术文章"), ("旅行时我通常优先坐公共交通", "用户旅行偏好公共交通")],
        "procedure": [("修改生产配置前必须先备份", "修改生产配置前必须先备份"), ("发送邮件前必须先给我看草稿", "发送邮件前必须展示草稿"), ("安装Python依赖时优先使用uv", "安装Python依赖优先使用uv"), ("推荐餐厅前必须先确认预算", "推荐餐厅前必须确认预算")],
    }
    for kind, items in typed.items():
        for index, (message, fact) in enumerate(items, 1):
            rows.append(case(f"type_{kind}_{index:03d}", "type_identification", message, required(f"fact_{kind}_{index}", kind, fact), query=fact))

    temporary = [("我今天有点头疼", "用户长期患有头痛"), ("昨晚没睡好现在很困", "用户长期失眠"), ("这两天临时不想吃辣", "用户一直不吃辣"), ("我现在正在等公交车", "用户长期在等公交车"), ("今天心情不太好", "用户长期情绪低落"), ("这周项目比较忙", "用户长期工作繁忙"), ("我可能晚点去健身", "用户固定晚间健身"), ("这次回答简短一点", "用户要求所有回答简短"), ("今天先不要提醒我", "用户永远不要提醒"), ("我这两天网络不太好", "用户长期网络异常")]
    for index, (message, forbidden) in enumerate(temporary, 1):
        rows.append(case(f"temporary_{index:03d}", "temporary_state", message, {"forbidden": [{"facts": [forbidden]}], "allowed_new_count": {"min": 0, "max": 1}}, query="用户有什么长期状态？"))

    conflicts = [
        ("喜欢周末爬山", "膝盖没有恢复，以后不考虑爬山", "以后不考虑爬山", "preference"),
        ("喜欢吃辣", "医生建议忌辣，以后不吃辣", "以后不吃辣", "preference"),
        ("主力电脑是Windows", "已经换成macOS作为主力电脑", "主力电脑是macOS", "profile"),
        ("目前住在合肥", "已经搬到桂林长期居住", "目前长期住在桂林", "profile"),
        ("正在准备考研", "已经考上并成为硕士研究生", "已经成为硕士研究生", "profile"),
        ("每天跑步", "膝盖受伤后已经暂停跑步", "目前暂停跑步", "preference"),
        ("主要阅读纸质书", "现在主要使用电子书", "现在主要使用电子书", "preference"),
        ("出行默认选择高铁", "现在长途出行更倾向飞机", "长途出行更倾向飞机", "preference"),
        ("回答越详细越好", "以后默认简洁回答", "以后默认简洁回答", "procedure"),
        ("邮件可以直接发送", "以后邮件必须先确认再发送", "邮件必须先确认再发送", "procedure"),
    ]
    for index, (old, message, new, kind) in enumerate(conflicts, 1):
        initial = [{"local_id": "old", "memory_type": kind, "summary": f"用户{old}", "status": "active"}]
        expected = required("new", kind, f"用户{new}")
        expected["expected_actions"] = [{"target_local_id": "old", "action": "supersede"}]
        expected["forbidden"] = [{"facts": [f"用户当前仍{old}"]}]
        rows.append(case(f"conflict_{index:03d}", "history_current_conflict", message, expected, initial=initial, source="personamem_adapted" if index <= 4 else "manual", query="用户当前的情况是什么？"))

    entities = [("我喜欢Python，但不喜欢Java", "用户喜欢Python", "用户喜欢Java"), ("我喜欢桂林米粉，但不喜欢螺蛳粉", "用户喜欢桂林米粉", "用户喜欢螺蛳粉"), ("工作电脑是Windows，个人电脑是macOS", "用户个人电脑是macOS", "用户工作电脑是macOS"), ("我住在桂林，父母住在合肥", "用户住在桂林", "用户住在合肥"), ("我对花生过敏，朋友对海鲜过敏", "用户对花生过敏", "用户对海鲜过敏"), ("我喜欢安静餐厅，伴侣喜欢热闹餐厅", "用户喜欢安静餐厅", "用户喜欢热闹餐厅"), ("北京行程预算三千，上海行程预算五千", "北京行程预算三千", "北京行程预算五千"), ("项目A使用SQLite，项目B使用PostgreSQL", "项目A使用SQLite", "项目A使用PostgreSQL")]
    for index, (message, fact, forbidden) in enumerate(entities, 1):
        expected = required("entity_fact", "profile" if index in {3,4,5} else "preference", fact)
        expected["forbidden"] = [{"facts": [forbidden]}]
        rows.append(case(f"entity_{index:03d}", "entity_attribute_conflict", message, expected, source="synthetic_bad_case", query=fact))

    noise = [("聊了天气和午饭后，我想补充：下个月搬到深圳", "用户下个月搬到深圳", "profile"), ("项目测试终于通过。顺便说，我以后看文档优先看中文", "用户优先看中文文档", "preference"), ("今天很热。以后修改数据库前一定先备份", "修改数据库前必须先备份", "procedure"), ("午饭吃了米粉，上周也完成了Agent项目上线", "用户完成了Agent项目上线", "event"), ("朋友喜欢跑步；我自己其实更喜欢游泳", "用户更喜欢游泳", "preference"), ("文章说大家爱咖啡，不过我一直更喜欢喝茶", "用户更喜欢喝茶", "preference"), ("先不聊代码，我已经从旧公司离职，正在找新工作", "用户正在找新工作", "profile"), ("天气不错。我的新手机已经换成安卓", "用户的新手机是安卓", "profile"), ("小王去了北京，而我上周去了上海出差", "用户上周去上海出差", "event"), ("今天开会很多。以后创建提醒前先和我确认时间", "创建提醒前必须确认时间", "procedure")]
    for index, (message, fact, kind) in enumerate(noise, 1):
        rows.append(case(f"noise_{index:03d}", "noise_extraction", message, required("key_fact", kind, fact), source="longmemeval_adapted" if index <= 4 else "manual", query=fact))

    negatives = [("如果我住在北京，你会推荐哪里", "用户住在北京"), ("小王喜欢爬山", "用户喜欢爬山"), ("文章里说程序员都爱咖啡", "用户喜欢咖啡"), ("你好，今天怎么样", "用户具有长期偏好"), ("请解释一下什么是SQLite", "用户使用SQLite"), ("有人说我可能喜欢跑步，但我自己不确定", "用户喜欢跑步")]
    for index, (message, forbidden) in enumerate(negatives, 1):
        rows.append(case(f"negative_{index:03d}", "negative", message, {"forbidden": [{"facts": [forbidden]}], "allowed_new_count": {"min": 0, "max": 0}}, source="synthetic_bad_case", query="用户有什么确定的长期信息？"))
    return rows


if __name__ == "__main__":
    target = Path(__file__).with_name("v1.jsonl")
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build()), encoding="utf-8")
    print(f"wrote {len(build())} cases to {target}")
