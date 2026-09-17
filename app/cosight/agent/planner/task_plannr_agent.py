from typing import Dict

from app.agent_dispatcher.infrastructure.entity.AgentInstance import AgentInstance
from app.cosight.agent.base.base_agent import BaseAgent
from app.cosight.agent.planner.prompt.planner_prompt import planner_system_prompt, \
    planner_create_plan_prompt, planner_re_plan_prompt, planner_finalize_plan_prompt
from app.cosight.llm.chat_llm import ChatLLM
from app.cosight.task.plan_report_manager import plan_report_event_manager
from app.cosight.task.task_manager import TaskManager
from app.cosight.tool.plan_toolkit import PlanToolkit
from app.cosight.tool.terminate_toolkit import TerminateToolkit


class TaskPlannerAgent(BaseAgent):
    def __init__(self, agent_instance: AgentInstance, llm: ChatLLM, plan_id, functions: Dict = None):
        self.plan = TaskManager.get_plan(plan_id)
        plan_toolkit = PlanToolkit(self.plan)
        terminate_toolkit = TerminateToolkit()
        all_functions = {"create_plan": plan_toolkit.create_plan, "update_plan": plan_toolkit.update_plan,
                         "terminate": terminate_toolkit.terminate}
        if functions:
            all_functions.update(functions)
        super().__init__(agent_instance, llm, all_functions, plan_id=plan_id)
        self.trace_type = "planner"

    def create_plan(self, question, output_format=""):
        self.history.append({"role": "system", "content": planner_system_prompt(question)})
        self.history.append({"role": "user", "content": planner_create_plan_prompt(question, output_format)})
        result = self.execute(
            self.history,
            max_iteration=1,
            tool_choice={"type": "function", "function": {"name": "create_plan"}},
        )
        return result

    def re_plan(self, question, output_format=""):
        self.history.append(
            {"role": "user", "content": planner_re_plan_prompt(question, self.plan.format(), output_format)})
        result = self.execute(self.history, max_iteration=1)
        return result

    def finalize_plan(self, question, output_format=""):
        self.history.append(
            {"role": "user", "content": planner_finalize_plan_prompt(question, self.plan.format(), output_format)})
        result = self.llm.chat_to_llm(self.history)
        self.plan.set_plan_result(result)
        plan_report_event_manager.publish("plan_result", self.plan)
        return f"""
Task:
{question}

Plan Status:
{self.plan.format()}

Summary:
{result}
"""
