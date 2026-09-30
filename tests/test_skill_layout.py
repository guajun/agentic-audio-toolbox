"""Layout/spec checks for the bundled skill (mirrors `gh skill publish` rules).

Standard library only. Guards that the skill stays installable and
self-contained: after `gh skill install` copies `skills/sam-audio/`, every
relative file reference in SKILL.md must still resolve inside that directory.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "sam-audio"
SKILL_MD = SKILL_DIR / "SKILL.md"

NAME_RULE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RELATIVE_REF = re.compile(r"(?<![\w/<>-])((?:scripts|references|assets)/[A-Za-z0-9_.\-/]+)")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    assert text.startswith("---"), "SKILL.md must start with frontmatter"
    _, front, body = text.split("---", 2)
    data: dict = {}
    current = None
    for line in front.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if value == "":
                current = {}
                data[key.strip()] = current
            else:
                current = None
                data[key.strip()] = value
        elif isinstance(current, dict):
            key, _, value = line.strip().partition(":")
            current[key.strip()] = value.strip().strip('"').strip("'")
    return data, body


class SkillLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SKILL_MD.read_text(encoding="utf-8")
        cls.front, cls.body = parse_frontmatter(cls.text)

    def test_name_matches_directory_and_spec_rules(self):
        name = self.front.get("name", "")
        self.assertEqual(name, SKILL_DIR.name)
        self.assertRegex(name, NAME_RULE)
        self.assertNotIn("--", name)
        self.assertLessEqual(len(name), 64)

    def test_description_present_and_bounded(self):
        description = self.front.get("description", "")
        self.assertTrue(description)
        self.assertLessEqual(len(description), 1024)
        for keyword in ("SAM Audio", "separat", "环境", "offline"):
            self.assertIn(keyword, description)

    def test_allowed_tools_is_string_if_present(self):
        if "allowed-tools" in self.front:
            self.assertIsInstance(self.front["allowed-tools"], str)

    def test_no_install_metadata_that_gh_strips(self):
        for line in self.text.splitlines():
            self.assertNotIn("metadata.github-", line)

    def test_relative_references_resolve_inside_skill_dir(self):
        body_refs = RELATIVE_REF.findall(self.body)
        self.assertTrue(body_refs, "expected bundled-file references in SKILL.md")
        for ref in body_refs:
            with self.subTest(ref=ref):
                self.assertTrue((SKILL_DIR / ref).is_file(), f"dangling reference: {ref}")

    def test_skill_is_self_contained(self):
        expected = [
            SKILL_DIR / "scripts" / "audio_toolbox.py",
            SKILL_DIR / "references" / "cli-reference.md",
        ]
        for path in expected:
            self.assertTrue(path.is_file(), f"missing bundled file: {path}")
        source = (SKILL_DIR / "scripts" / "audio_toolbox.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "from torch", "import sam_audio", "from sam_audio",
                         "import requests", "urllib.request", "socket."):
            self.assertNotIn(forbidden, source)

    def test_frontmatter_declares_provenance_and_licenses(self):
        metadata = self.front.get("metadata", {})
        self.assertEqual(metadata.get("sam-upstream-commit"),
                         "c603de8794cc16880dc01be0f1e868f6c2845417")
        self.assertIn("Meta SAM License", self.text)
        self.assertIn("MIT", self.text)


if __name__ == "__main__":
    unittest.main()