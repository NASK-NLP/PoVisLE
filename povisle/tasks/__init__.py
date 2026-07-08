from povisle.tasks.base import BaseTask, ParsingMethod, ParsingStatus, ScoringMethod


def get_tasks(task_names: list[str] | None) -> list[BaseTask]:
    if not task_names:
        return []

    all_task_names = ["mcq", "yn", "open"]
    if "all" in task_names:
        task_names = all_task_names

    return [BaseTask.from_name(name) for name in task_names]
