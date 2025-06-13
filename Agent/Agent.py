#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

import os
import shutil
import logging
import subprocess
import time
import openai  # 需要安装openai包

# # start LLMs' Agent

# # Copy the project into the cache
# # 所有agnet记录都存在logs.log文件
# 0. 因为解析{file_path}文件失败，现在需要使用Agent自动化处理。现在需要做的是使用强大的大模型推理能力，实时覆盖重写模块。
# 核心的流程是将项目复制到cache里面，然后定位本文件的所有模块，依次读入让大模型理解如何根据现有的代码标准和大模型本身扩展能力进行模块重新。
# 模块重写的标准是不改变原有模块的内容，根据原模块的模板进行扩写，然后写入注册表。写完对应的py文件之后再cache对应模块进行覆盖保存。只针对当前文件对应模块的内容进行理解推理和重写
# 然后llm启动cache里面的app.py文件，检测数据流是否正确，如果正确就将cache模块覆盖本地模块，然后启动自动加载的函数，再检测数据流。所有执行记录都在logs.log文件里面。
# 大模型以logs.log文件为数据中心，理解，重写输出到一个变量，然后保存该文件，然后启动测试，测试数据记录到logs.log文件里里面，然后再读取数据流是否成功，如果成就输入确定型号，然后执行覆盖操作
# 如果不成功，就记录出错原因，然后清零logs.log文件，出错原因写入，再执行遍历(记录)，再执行上述操作

# 1. 定位运行项目路径
# 2. 讲项目的所有文件夹包括文件复制到 项目目录/cache下 (cache不复制)
# 3. 首先定位该模块
# 2. 然后遍历该模块所有目录文件
# 3. 依次读入每个文件下的py文件
# 4. prompt 
class CodeAutoAgent:
    def __init__(self, file_path):
        self.original_project = os.path.dirname(os.path.abspath(file_path))
        self.cache_dir = os.path.join(self.original_project, "cache")
        self.log_file = os.path.join(self.original_project, "logs.log")
        self.current_module = os.path.basename(file_path)
        self.max_retries = 3
        
        # 配置日志记录
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def clone_project(self):
        """将项目复制到cache目录"""
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            
            ignore = shutil.ignore_patterns('cache', 'logs.log', '__pycache__')
            shutil.copytree(self.original_project, self.cache_dir, ignore=ignore)
            logging.info(f"Project cloned to {self.cache_dir}")
            return True
        except Exception as e:
            logging.error(f"Clone failed: {str(e)}")
            return False

    def process_modules(self):
        """处理所有模块"""
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith(".py"):
                    self._rewrite_module(os.path.join(root, file))

    def _rewrite_module(self, file_path):
        """使用LLM重写单个模块"""
        try:
            with open(file_path, "r+") as f:
                original_code = f.read()
                
                # 构造LLM提示
                prompt = f"""根据现有代码标准扩展此模块，保持原有功能不变。只添加新功能，不要修改现有代码。
                原代码：
                {original_code}
                
                请返回完整的重写后的Python代码，只需要代码内容，不要包含解释。"""
                
                # 调用LLM API
                rewritten_code = self._call_llm_api(prompt)
                
                # 验证代码有效性
                if self._validate_code(rewritten_code):
                    f.seek(0)
                    f.write(rewritten_code)
                    f.truncate()
                    logging.info(f"Rewrote {os.path.basename(file_path)}")
                else:
                    raise ValueError("Generated code validation failed")
        except Exception as e:
            logging.error(f"Rewrite failed for {file_path}: {str(e)}")
            raise

    def _call_llm_api(self, prompt):
        """调用大模型API（示例使用OpenAI）"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一个资深Python开发者，擅长模块化代码重构和功能扩展"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        return response.choices[0].message['content'].strip()

    def _validate_code(self, code):
        """基础代码验证"""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error in generated code: {e}")
            return False

    def test_application(self):
        """测试缓存中的应用程序"""
        app_path = os.path.join(self.cache_dir, "app.py")
        try:
            result = subprocess.run(
                ["python", app_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            with open(self.log_file, "a") as f:
                f.write("\n=== TEST OUTPUT ===\n")
                f.write(result.stdout)
                f.write("\n=== TEST ERRORS ===\n")
                f.write(result.stderr)
            
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Test failed: {str(e)}")
            return False

    def deploy_changes(self):
        """部署成功修改"""
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(
                    self.original_project,
                    os.path.relpath(src_path, self.cache_dir)
                )
                shutil.copy2(src_path, dest_path)
        logging.info("Changes deployed successfully")

    def cleanup(self):
        """清理日志"""
        open(self.log_file, "w").close()

    def execute_workflow(self):
        """执行完整工作流程"""
        for attempt in range(self.max_retries):
            logging.info(f"Attempt {attempt + 1}/{self.max_retries}")
            
            if not self.clone_project():
                continue
                
            try:
                self.process_modules()
                if self.test_application():
                    self.deploy_changes()
                    logging.info("Process completed successfully")
                    return True
            except Exception as e:
                logging.error(f"Workflow error: {str(e)}")
            
            self.cleanup()
            time.sleep(2 ** attempt)  # 指数退避
        
        logging.error("All attempts failed")
        return False

if __name__ == "__main__":
    # 使用示例
    agent = CodeAutoAgent(__file__)  # 传入当前文件路径
    if agent.execute_workflow():
        print("Process completed successfully")
    else:
        print("Process failed. Check logs.log for details")