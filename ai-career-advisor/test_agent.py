# -*- coding: utf-8 -*-
"""测试AI职业顾问Agent"""
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from agent.career_advisor_agent import CareerAdvisorAgent

def main():
    print("="*60)
    print("AI职业顾问 - 交互式测试")
    print("="*60)
    print("\n提示: 输入 'quit' 或 'exit' 退出")
    print("示例问题:")
    print("- 我会Python和SQL,能拿多少薪水?")
    print("- 我想学习新技能,有什么建议?")
    print("- 推荐适合我的职位")
    print("-" * 60)
    
    # 创建Agent
    agent = CareerAdvisorAgent()
    
    # 欢迎消息
    print("\nAgent:", agent.chat("你好"))
    
    # 交互循环
    while True:
        print("\n" + "-" * 60)
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出', 'q']:
            print("\n感谢使用AI职业顾问! 再见!")
            break
        
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\nAgent:\n{response}")

if __name__ == "__main__":
    main()
