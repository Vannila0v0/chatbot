from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = ROOT / "skills" / "agent-interviewer"
SKILL_FILE = SKILL_DIR / "SKILL.md"
PATTERNS_FILE = SKILL_DIR / "references" / "interview-patterns.md"
RUBRIC_FILE = SKILL_DIR / "references" / "evaluation-rubric.md"
PROFILE_TEMPLATE = SKILL_DIR / "templates" / "candidate-profile.md"
PERSONAL_PROFILE = ROOT / "docs" / "interview" / "candidate-profile.md"
SKILLS_INDEX = ROOT / "skills" / "README.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_files_exist():
    required = [
        SKILL_FILE,
        PATTERNS_FILE,
        RUBRIC_FILE,
        PROFILE_TEMPLATE,
        PERSONAL_PROFILE,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Missing required files: {missing}"


def test_frontmatter_is_discoverable():
    text = read(SKILL_FILE)
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    frontmatter = match.group(1)
    assert re.search(r"^name:\s*agent-interviewer$", frontmatter, flags=re.MULTILINE)
    description = re.search(r"^description:\s*(.+)$", frontmatter, flags=re.MULTILINE)
    assert description
    assert description.group(1).startswith("Use when ")
    assert len(frontmatter) <= 1024


def test_frontmatter_has_akashic_metadata():
    text = read(SKILL_FILE)
    frontmatter = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL).group(1)
    metadata = re.search(r"^metadata:\s*(\{.+\})$", frontmatter, flags=re.MULTILINE)
    assert metadata, "frontmatter must include metadata.akashic"
    parsed = json.loads(metadata.group(1))
    assert parsed["akashic"]["emoji"]


def test_skill_is_registered_in_builtin_index():
    text = read(SKILLS_INDEX)
    assert "`agent-interviewer`" in text
    assert "skills/agent-interviewer/SKILL.md" in text


def test_skill_contains_hard_behavioral_contracts():
    text = read(SKILL_FILE)
    markers = {
        "one question": "<!-- contract:one-question -->",
        "adaptive follow-up": "<!-- contract:adaptive-follow-up -->",
        "resume priority": "<!-- contract:resume-priority -->",
        "deferred feedback": "<!-- contract:deferred-feedback -->",
        "evidence review": "<!-- contract:evidence-review -->",
    }
    missing = [label for label, marker in markers.items() if marker not in text]
    assert not missing, f"Missing behavioral contracts: {missing}"

    for state in [
        "Intake",
        "Opening",
        "Project Deep Dive",
        "Domain Depth",
        "Engineering & Foundations",
        "Candidate Questions",
        "Review",
    ]:
        assert state in text


def test_references_are_routed_from_skill():
    text = read(SKILL_FILE)
    assert "references/interview-patterns.md" in text
    assert "references/evaluation-rubric.md" in text
    assert "templates/candidate-profile.md" in text
    assert "docs/interview/candidate-profile.md" in text


def test_interview_patterns_have_dated_sources_and_caveats():
    text = read(PATTERNS_FILE)
    assert "2026-07-06" in text
    for url in [
        "https://notes.kamacoder.com/interview/llm/",
        "https://cj.sina.com.cn/articles/view/7879848900/1d5acf3c401902rm12",
        "https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide",
    ]:
        assert url in text
    assert "启发式" in text
    assert "不是统计" in text
    assert "登录" in text


def test_rubric_distinguishes_weak_evidence_from_uncovered():
    text = read(RUBRIC_FILE)
    assert "未覆盖" in text
    assert "弱证据" in text
    assert "未覆盖不等于能力弱" in text
    assert "通过概率" in text
    assert "区间" in text
    assert "背稿" in text
    assert "表达不流畅" in text


def test_profile_template_has_required_sections():
    text = read(PROFILE_TEMPLATE)
    sections = [
        "目标岗位与级别",
        "面试设置",
        "简历摘要",
        "项目声明",
        "个人职责",
        "规模与指标",
        "失败案例",
        "技术取舍",
        "目标 JD",
        "已知短板",
        "隐私",
    ]
    missing = [section for section in sections if section not in text]
    assert not missing, f"Missing profile sections: {missing}"


def test_personal_profile_marks_unverified_claims():
    text = read(PERSONAL_PROFILE)
    assert "待本人确认" in text
    assert "Agent Runtime" in text
    assert all(term in text for term in ["Proactive", "Drift", "ToolRegistry", "MCP"])
