from datetime import datetime
from app.cosight.agent.actor.instance.actor_agent_instance import create_actor_instance
from llm import llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision
from app.cosight.task.plan_report_manager import plan_report_event_manager


import os
import time
from threading import Thread
import shutil
import re

from app.cosight.agent.actor.task_actor_agent import TaskActorAgent
from app.cosight.agent.planner.instance.planner_agent_instance import create_planner_instance
from app.cosight.agent.planner.task_plannr_agent import TaskPlannerAgent
from app.cosight.task.task_manager import TaskManager
from app.cosight.task.todolist import Plan
from app.cosight.task.time_record_util import time_record
from app.cosight.task.trajectory_tracer import create_trajectory_tracer, remove_trajectory_tracer
from app.common.logger_util import logger

import json


class CogResearch:
    def __init__(self, plan_llm, act_llm, tool_llm, vision_llm, work_space_path: str = None, message_uuid: str|None = None, trace_dir: str = None):
        self.work_space_path = work_space_path or os.getenv("WORKSPACE_PATH") or os.getcwd()
        self.plan_id = message_uuid if message_uuid else f"plan_{int(time.time())}"
        self.plan = Plan(work_space_path=self.work_space_path)
        TaskManager.set_plan(self.plan_id, self.plan)
        self.trace_dir = trace_dir or os.getenv("COSIGHT_TRACE_DIR") or os.path.join(self.work_space_path, "traces")
        self.trajectory_tracer = create_trajectory_tracer(self.plan_id, self.trace_dir)
        
        # 设置Langfuse追踪上下文
        # 使用plan_id作为session_id，让整个任务的所有traces都关联在一起
        # trace_id会自动生成，每个Agent调用都是独立的trace
        # 这样在Langfuse中可以看到完整的session replay
        plan_llm.set_trace_context(
            trace_id=None,  # 不指定trace_id，让每次调用自动生成
            session_id=self.plan_id,  # 使用plan_id作为session_id
            tags=["planning"],
            metadata={"agent_type": "planner", "plan_id": self.plan_id}
        )
        act_llm.set_trace_context(
            trace_id=None,
            session_id=self.plan_id,
            tags=["execution"],
            metadata={"agent_type": "actor", "plan_id": self.plan_id}
        )
        tool_llm.set_trace_context(
            trace_id=None,
            session_id=self.plan_id,
            tags=["tool"],
            metadata={"agent_type": "tool", "plan_id": self.plan_id}
        )
        vision_llm.set_trace_context(
            trace_id=None,
            session_id=self.plan_id,
            tags=["vision"],
            metadata={"agent_type": "vision", "plan_id": self.plan_id}
        )
        
        self.task_planner_agent = TaskPlannerAgent(create_planner_instance("task_planner_agent"), plan_llm, self.plan_id)
        self.act_llm = act_llm  # Store llm for later use
        self.tool_llm = tool_llm
        self.vision_llm = vision_llm

    @time_record
    def execute(self, question, attached_files=[], output_format=""):
        """
            question: 输入的文本问题（中文/英文）
            attached_files: 附件列表, 最多50个, 可以为空列表
            output_format: 指定的输出报告文件类型, 支持 md, pdf, word 等
        """
        # 如果有附件，先处理附件。将附件的路径包装进question中
        if attached_files:
            processed_files = []
            for file_path in attached_files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.abspath(os.path.join(self.work_space_path, file_name))
                    shutil.copy2(file_path, dest_path)

                    processed_files.append({
                        "name": file_name,
                        "path": dest_path,
                    })
                else:
                    logger.warning(f"Attached file does not exist: {file_path}")

            file_list_str = "\n".join(
                f"- {item['name']}: {item['path']}"
                for item in processed_files
            )

            attachment_msg = (
                "Attached files have been copied to the current task workspace.\n"
                "IMPORTANT: When reading files, you MUST use the absolute paths below. "
                "Do NOT assume files are in os.getcwd().\n\n"
                f"{file_list_str}\n\n"
                f"Current task workspace: {os.path.abspath(self.work_space_path)}\n\n"
            )

            question = attachment_msg + question

        # 更新所有LLM的metadata，添加任务信息
        task_metadata = {
            "task_question": question,
            "plan_id": self.plan_id
        }
        
        for llm in [self.task_planner_agent.llm, self.act_llm, self.tool_llm, self.vision_llm]:
            if hasattr(llm, 'current_metadata'):
                llm.current_metadata.update(task_metadata)
        
        create_task = question
        retry_count = 0
        while not self.plan.get_ready_steps() and retry_count < 3:
            create_result = self.task_planner_agent.create_plan(create_task, output_format)
            create_task += f"\nThe plan creation result is: {create_result}\nCreation failed, please carefully review the plan creation rules and select the create_plan tool to create the plan"
            retry_count += 1
        
        # 使用持续监控的方式，而不是等待所有步骤完成
        active_threads = {}  # 存储活跃的线程 {step_index: thread}
        
        while True:
            # 检查是否有新的可执行步骤
            ready_steps = self.plan.get_ready_steps()
            
            # 启动新的可执行步骤
            for step_index in ready_steps:
                if step_index not in active_threads:
                    if not self.plan.claim_step(step_index):
                        continue
                    logger.info(f"Starting new step {step_index}")
                    thread = Thread(target=self._execute_single_step, args=(question, step_index))
                    thread.daemon = True
                    thread.start()
                    active_threads[step_index] = thread
            
            # 检查已完成的线程
            completed_steps = []
            for step_index, thread in active_threads.items():
                if not thread.is_alive():
                    completed_steps.append(step_index)
            
            # 移除已完成的线程
            for step_index in completed_steps:
                del active_threads[step_index]
                logger.info(f"Step {step_index} completed and thread removed")
            
            # 如果没有活跃线程且没有可执行步骤，则退出
            if not active_threads and not ready_steps:
                logger.info("No more ready steps to execute and no active threads")
                break
            
            # 短暂休眠，避免CPU占用过高
            import time
            time.sleep(0.1)
        
        final_result = self.task_planner_agent.finalize_plan(question, output_format)
        self.trajectory_tracer.dump()
        remove_trajectory_tracer(self.plan_id)
        return final_result

    def _execute_single_step(self, question, step_index):
        """执行单个步骤"""
        try:
            logger.info(f"Starting execution of step {step_index}")
            # 每个线程创建独立的TaskActorAgent实例
            task_actor_agent = TaskActorAgent(
                create_actor_instance(f"actor_for_step_{step_index}", self.work_space_path),
                self.act_llm,
                self.vision_llm,
                self.tool_llm,
                self.plan_id,
                work_space_path=self.work_space_path
            )
            result = task_actor_agent.act(question=question, step_index=step_index)
            logger.info(f"Completed execution of step {step_index} with result: {result}")
        except Exception as e:
            logger.error(f"Error executing step {step_index}: {e}", exc_info=True)
            try:
                self.plan.mark_step(step_index, step_status="blocked", step_notes=str(e))
                plan_report_event_manager.publish("plan_process", self.plan)
            except Exception:
                logger.error(f"Failed to mark step {step_index} blocked after execution error", exc_info=True)

    def execute_steps(self, question, ready_steps):
        from threading import Thread, Semaphore
        from queue import Queue

        results = {}
        result_queue = Queue()
        semaphore = Semaphore(min(5, len(ready_steps)))

        def execute_step(step_index):
            semaphore.acquire()
            try:
                logger.info(f"Starting execution of step {step_index}")
                # 每个线程创建独立的TaskActorAgent实例
                task_actor_agent = TaskActorAgent(
                    create_actor_instance(f"actor_for_step_{step_index}", self.work_space_path),
                    self.act_llm,
                    self.vision_llm,
                    self.tool_llm,
                    self.plan_id,
                    work_space_path=self.work_space_path
                )
                result = task_actor_agent.act(question=question, step_index=step_index)
                logger.info(f"Completed execution of step {step_index} with result: {result}")
                result_queue.put((step_index, result))
            finally:
                semaphore.release()

        # 为每个ready_step创建并执行线程
        threads = []
        for step_index in ready_steps:
            thread = Thread(target=execute_step, args=(step_index,))
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 收集结果
        while not result_queue.empty():
            step_index, result = result_queue.get()
            results[step_index] = result

        return results


# if __name__ == '__main__':
#     # 配置工作区
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     # 获取当前时间并格式化
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     # 构造路径：/xxx/xxx/work_space/work_space_时间戳
#     work_space_path = os.path.join(BASE_DIR, 'work_space', f'work_space_{timestamp}')
#     os.makedirs(work_space_path, exist_ok=True)

#     # 配置CoSight
#     cosight = CogResearch(llm_for_plan, llm_for_act, llm_for_tool, llm_for_vision, work_space_path)

#     # 运行CoSight
#     # result = cosight.execute("帮我写一篇中兴通讯的分析报告")
#     result = cosight.execute(
#         question="Please help me generate a multimodal report with both text and images based on the topic: Language-based AI systems have grown rapidly in recent years",
#         attached_files=[],
#         output_format="markdown"
#     )
#     logger.info(f"final result is {result}")


if __name__ == '__main__':
    # 配置基础路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 获取当前时间并格式化，用于区分批次
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 定义输入/输出文件路径与图片基础目录
    quiz_file_path = os.getenv(
        "QUIZ_FILE_PATH",
        '/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/data/quiz.jsonl'
    )
    image_base_dir = os.getenv(
        "QUIZ_IMAGE_BASE_DIR",
        '/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/data/images/'
    )
    trace_dir = os.getenv(
        "COSIGHT_TRACE_DIR",
        "/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/trace_mmdeepresearch"
    )
    output_file_path = os.path.join(BASE_DIR, f'quiz_results_{timestamp}.jsonl')
    
    if not os.path.exists(quiz_file_path):
        logger.error(f"找不到任务文件: {quiz_file_path}")
    else:
        logger.info(f"开始批量处理，结果将实时保存至: {output_file_path}")
        
        with open(quiz_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 1. 解析当前任务数据
                    task_data = json.loads(line)
                    task_id = task_data.get('id', f'unknown_{line_num}')
                    body = task_data.get('body', '')
                    caption = task_data.get('caption', '')
                    image_filenames = task_data.get('image_url')
                    include_caption = os.getenv("COSIGHT_INCLUDE_CAPTION", "false").lower() in {"1", "true", "yes", "on"}
                    
                    # 2. 组装 prompt 与绝对图片路径
                    is_chinese = bool(re.search(r"[\u4e00-\u9fff]", body)) if body else True
                    question_body = f"[{caption}]\n{body}" if include_caption and caption else body
                    if is_chinese:
                        question = f"{question_body} 如果用户未对最终输出提出任何具体要求，你通常需要生成一份最终的 Markdown 报告作为响应。"
                    else:
                        question = f"{question_body} If the user has not specified any particular requirements for the final output, you usually need to generate a final markdown report in response."
                    attached_files = [os.path.join(image_base_dir, img) for img in image_filenames]
                    
                    logger.info(f"========== 开始执行任务 ID: {task_id} ==========")
                    
                    # 3. 为当前任务创建独立工作区
                    task_work_space = os.path.join(BASE_DIR, 'work_space', f'work_space_{timestamp}', f'task_{task_id}')
                    os.makedirs(task_work_space, exist_ok=True)
                    
                    # 4. 重新实例化 CoSight，确保 Plan 和 Trace 上下文是干净的
                    # 使用唯一的 message_uuid 避免 TaskManager 的 plan 冲突
                    cosight = CogResearch(
                        plan_llm=llm_for_plan, 
                        act_llm=llm_for_act, 
                        tool_llm=llm_for_tool, 
                        vision_llm=llm_for_vision, 
                        work_space_path=task_work_space,
                        message_uuid=f"plan_{task_id}_{int(time.time())}",
                        trace_dir=trace_dir
                    )
                    
                    # 5. 执行智能体主流程
                    result = cosight.execute(
                        question=question,
                        attached_files=attached_files,
                        output_format="markdown"
                    )

                    # 增量保存结果（成功/失败都写入 jsonl，方便在 Slurm 日志较少时排查进度）
                    output_data = task_data.copy()
                    if "image_url" not in output_data and "images" in output_data:
                        output_data["image_url"] = output_data.pop("images")
                    output_data['cosight_result'] = result
                    with open(output_file_path, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    
                    logger.info(f"任务 ID: {task_id} 执行完成。")
                        
                except Exception as e:
                    # 捕获单个任务的异常，防止中断整个评估流程
                    logger.error(f"执行任务 ID {task_id} 时发生严重错误: {e}", exc_info=True)
                    
                    # 记录失败状态到文件
                    output_data = task_data.copy()
                    if "image_url" not in output_data and "images" in output_data:
                        output_data["image_url"] = output_data.pop("images")
                    output_data['cosight_error'] = str(e)
                    with open(output_file_path, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        
        logger.info(f"========== 批量测试全部结束 ==========")

# if __name__ == '__main__':
#     # 配置基础路径
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
#     # 获取当前时间并格式化，用于区分批次
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
#     # 定义输入/输出文件路径与图片基础目录
#     quiz_file_path = '/home/export/base/ycsc_chenkh/hitici_07/online1/Cog-Research/topics.jsonl'
#     output_file_path = os.path.join(BASE_DIR, f'quiz_results_{timestamp}.jsonl')
    
#     if not os.path.exists(quiz_file_path):
#         logger.error(f"找不到任务文件: {quiz_file_path}")
#     else:
#         logger.info(f"开始批量处理，结果将实时保存至: {output_file_path}")
        
#         with open(quiz_file_path, 'r', encoding='utf-8') as f:
#             for line_num, line in enumerate(f, 1):
#                 line = line.strip()
#                 if not line:
#                     continue
                
#                 try:
#                     # 1. 解析当前任务数据
#                     task_data = json.loads(line)
#                     task_id = task_data.get('id', f'unknown_{line_num}')
#                     scope = task_data.get('scope', '')
#                     topic = task_data.get('topic', '')

#                     instruction_prompt = (
#                         "Please conduct a comprehensive investigation on the topic provided below. "
#                         "Your goal is to generate a highly structured, richly illustrated (multimodal) markdown research report. "
#                         "Seamlessly integrate visual evidence (e.g., charts, graphs) to support your findings."
#                         ""
#                     )
                    
#                     # 2. 组装 prompt 与绝对图片路径
#                     question = f"{instruction_prompt}\n\n[Scope]: {scope}\n[Topic]: {topic}"
#                     attached_files = []
                    
#                     logger.info(f"========== 开始执行任务 ID: {task_id} ==========")
                    
#                     # 3. 为当前任务创建独立工作区
#                     task_work_space = os.path.join(BASE_DIR, 'work_space', f'work_space_{timestamp}', f'task_{task_id}')
#                     os.makedirs(task_work_space, exist_ok=True)
                    
#                     # 4. 重新实例化 CoSight，确保 Plan 和 Trace 上下文是干净的
#                     # 使用唯一的 message_uuid 避免 TaskManager 的 plan 冲突
#                     cosight = CogResearch(
#                         plan_llm=llm_for_plan, 
#                         act_llm=llm_for_act, 
#                         tool_llm=llm_for_tool, 
#                         vision_llm=llm_for_vision, 
#                         work_space_path=task_work_space,
#                         message_uuid=f"plan_{task_id}_{int(time.time())}"
#                     )
                    
#                     # 5. 执行智能体主流程
#                     result = cosight.execute(
#                         question=question,
#                         attached_files=attached_files,
#                         output_format="markdown"
#                     )
                    
#                     logger.info(f"任务 ID: {task_id} 执行完成。")
                    
#                     # 6. 增量保存结果
#                     output_data = task_data.copy()
#                     output_data['cosight_result'] = result
#                     with open(output_file_path, 'a', encoding='utf-8') as out_f:
#                         out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        
#                 except Exception as e:
#                     # 捕获单个任务的异常，防止中断整个评估流程
#                     logger.error(f"执行任务 ID {task_id} 时发生严重错误: {e}", exc_info=True)
                    
#                     # 记录失败状态到文件
#                     output_data = task_data.copy()
#                     output_data['cosight_error'] = str(e)
#                     with open(output_file_path, 'a', encoding='utf-8') as out_f:
#                         out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        
#         logger.info(f"========== 批量测试全部结束 ==========")
