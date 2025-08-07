from typing import Optional


class DeepSeekR1ReasoningParser:
    """Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.

    See details at:
    https://github.com/vllm-project/vllm/blob/main/vllm/reasoning/deepseek_r1_reasoning_parser.py
    """

    start_token: str = "<think>"
    end_token: str = "</think>"

    def extract_reasoning_content(self, model_output: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text <think>abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = model_output_parts[2] if model_output_parts[1] else model_output_parts[0]

        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.end_token not in model_output:
            model_output = model_output.strip()
            return None, model_output
        else:
            reasoning_content, _, content = model_output.partition(self.end_token)
            # If the end token is not found, return the model output as is.
            # It should not happen since we already checked for the presence
            # of the end token.
            # If generation stops right after end-of-think, return null content
            final_content = content or None
            reasoning_content = reasoning_content.strip()
            final_content = final_content.strip()
            return reasoning_content, final_content
