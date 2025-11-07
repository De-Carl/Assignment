# -*- coding: utf-8 -*-
"""
Flask Web应用主文件
提供AI职业顾问的Web界面
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from agent.career_advisor_agent import CareerAdvisorAgent

app = Flask(__name__)
agent = None

def init_agent():
    """初始化智能体"""
    global agent
    try:
        agent = CareerAdvisorAgent()
        print('✓ AI职业顾问Agent已加载')
        return True
    except Exception as e:
        print(f'✗ Agent加载失败: {e}')
        return False

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '消息不能为空'}), 400
        
        # 获取AI回复
        response = agent.chat(user_message)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'agent_loaded': agent is not None
    })

if __name__ == '__main__':
    print('='*60)
    print('AI职业顾问 Web应用')
    print('='*60)
    
    # 初始化智能体
    if init_agent():
        print('\n启动服务器...')
        print('访问地址: http://127.0.0.1:5000')
        print('按 Ctrl+C 停止服务器')
        print('='*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print('\n✗ Agent初始化失败,请先训练模型')
        print('运行: python train_models.py')
