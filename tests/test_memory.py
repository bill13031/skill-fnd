from fake_news_skillrl.memory import SkillsOnlyMemory


def test_template_retrieval_and_formatting():
    memory = SkillsOnlyMemory("memory_data/fake_news/claude_style_skills.json")
    retrieved = memory.retrieve("Verify whether archive footage is being reused in a current event post.", top_k=2)
    prompt = memory.format_for_prompt(retrieved)
    assert retrieved["task_type"] in {"out_of_context", "temporal_inconsistency", "misleading_caption"}
    assert "General Principles" in prompt
    assert "Mistakes To Avoid" in prompt
