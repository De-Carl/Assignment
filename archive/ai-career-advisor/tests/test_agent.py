# -*- coding: utf-8 -*-
"""
智能体测试脚本
测试AI职业顾问Agent的功能
"""

import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import pytest
from agent.career_advisor_agent import CareerAdvisorAgent

def test_agent_creation():
    """测试智能体创建"""
    agent = CareerAdvisorAgent()
    assert agent is not None
    print("✓ 智能体创建成功")

def test_chat_hello():
    """测试基本对话"""
    agent = CareerAdvisorAgent()
    response = agent.chat("你好")
    assert response is not None
    assert len(response) > 0
    print(f"✓ 对话测试通过: {response[:50]}...")

def test_chat_skills():
    """测试技能查询"""
    agent = CareerAdvisorAgent()
    response = agent.chat("我会Python和SQL")
    assert response is not None
    print(f"✓ 技能查询测试通过")

if __name__ == "__main__":
    print("开始测试AI职业顾问Agent...")
    test_agent_creation()
    test_chat_hello()
    test_chat_skills()
    print("\n所有测试完成!")
