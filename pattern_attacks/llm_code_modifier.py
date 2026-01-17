import os
from openai import OpenAI
from typing import Dict, List
import time

# ==================== Pattern Descriptions ====================

PATTERN_DESCRIPTIONS = {
    'docstrings': {
        1: "Add a minimal one-line docstring describing the function's purpose",
        2: "Add a standard docstring with sections for: summary, Args (parameter descriptions), and Returns",
        3: "Add a detailed docstring with: summary, detailed description, Args with type information, Returns with details, and Examples section with usage demonstrations",
    },
    'comments': {
        1: "Add only a single function-level comment at the top explaining the overall purpose",
        2: "Add block-level comments before major code sections (loops, conditionals, returns) explaining their purpose",
        3: "Add detailed comments including: function-level comment, block-level comments, and inline comments for key operations or non-obvious logic",
    },
    'type_hints': {
        1: "Add basic type hints to function signature using simple types (str, int, bool, list, dict, float)",
        2: "Add standard type hints including Optional, Union, and proper collection types (List[str], Dict[str, int])",
        3: "Add comprehensive type hints including: function signature with complex types, type hints for important internal variables, and use of advanced typing features when appropriate",
    },
    'error_handling': {
        1: "Add basic input validation at the beginning of the function to check parameter types and values",
        2: "Add standard error handling with try-except blocks using specific exception types where appropriate",
        3: "Add comprehensive error handling including: input validation with descriptive error messages, try-except blocks with specific exceptions, and helpful error context",
    },
    'variable_style': {
        1: "Use descriptive variable names that clearly indicate their purpose",
        2: "Use descriptive variable names AND break complex expressions into intermediate variables for clarity",
        3: "Use highly descriptive variable names AND extensively use intermediate variables to make every step explicit and readable",
    },
}

# ==================== Prompts ====================

SYSTEM_PROMPT = """You are a code refactoring assistant. Your task is to modify Python code according to specified patterns while preserving the exact original logic and functionality.

Rules:
1. NEVER change the code's logic or behavior
2. Output ONLY the modified code without any explanations or markdown formatting
3. Maintain syntactic correctness and ensure the code runs identically to the original
4. Follow the specified level for each pattern exactly as described
5. For patterns not mentioned, keep the original code style unchanged"""

USER_PROMPT_TEMPLATE = """Modify the following Python code according to these specifications:

**Pattern Modifications:**
{pattern_instructions}

**Original Code:**
```python
{original_code}
```

**Output the modified code below (code only, no explanations):**"""

# ==================== Helper Functions ====================

def generate_pattern_instructions(config: Dict[str, int]) -> str:
    """
    Generate instruction text based on config.
    Only includes patterns with level > 0.
    
    Args:
        config: Dictionary with pattern names as keys and levels (0-3) as values
                Example: {'docstrings': 2, 'comments': 1, 'type_hints': 0}
                Level 0 means no modification for that pattern
    
    Returns:
        Formatted instruction string
    """
    instructions = []
    
    # Process in consistent order
    for pattern_name in ['docstrings', 'comments', 'type_hints', 'error_handling', 'variable_style']:
        level = config.get(pattern_name, 0)
        
        # Skip if level is 0 (no modification)
        if level == 0:
            continue
        
        if level not in PATTERN_DESCRIPTIONS[pattern_name]:
            raise ValueError(f"Invalid level {level} for pattern {pattern_name}. Must be 0-3.")
        
        description = PATTERN_DESCRIPTIONS[pattern_name][level]
        pattern_display_name = pattern_name.replace('_', ' ').title()
        instructions.append(f"- {pattern_display_name} (Level {level}): {description}")
    
    if not instructions:
        return "- No modifications required (keep original code as-is)"
    
    return "\n".join(instructions)


def clean_code_output(code: str) -> str:
    """
    Clean up the LLM output to extract pure code.
    
    Args:
        code: Raw output from LLM
    
    Returns:
        Cleaned code string
    """
    code = code.strip()
    
    # Remove markdown code blocks if present
    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
        code = code.split("```", 1)[0]
    elif code.startswith("```"):
        code = code.split("```", 1)[1]
        code = code.split("```", 1)[0]
    
    return code.strip()


# ==================== Main Functions ====================

def modify_code_with_deepseek(
    code: str,
    config: Dict[str, int],
    api_key: str = None,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_retries: int = 3
) -> str:
    """
    Modify a single code snippet using DeepSeek API.
    
    Multiple patterns in config will be applied TOGETHER in ONE API call.
    
    Args:
        code: Original Python code
        config: Pattern configuration with multiple patterns
                Example: {
                    'docstrings': 2,      # Will apply
                    'comments': 1,        # Will apply
                    'type_hints': 1,      # Will apply
                    'error_handling': 0,  # Will NOT apply (level 0)
                    'variable_style': 0   # Will NOT apply (level 0)
                }
                All patterns with level > 0 are applied TOGETHER in one call
        api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env variable)
        model: Model name ("deepseek-chat" or "deepseek-coder")
        temperature: Sampling temperature (0.0-1.0)
        max_retries: Maximum number of retries on failure
    
    Returns:
        Modified code string with ALL specified patterns applied
    """
    # Get API key
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided or set in DEEPSEEK_API_KEY environment variable")
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # Generate instructions for ALL patterns in config
    pattern_instructions = generate_pattern_instructions(config)
    
    # Create prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        pattern_instructions=pattern_instructions,
        original_code=code.strip()
    )
    
    # Call API with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=4096
            )
            
            # Extract and clean output
            modified_code = response.choices[0].message.content
            modified_code = clean_code_output(modified_code)
            
            return modified_code
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed: {e}")
                return code  # Return original code on failure


def modify_codes_batch(
    codes: List[str],
    config: Dict[str, int],
    api_key: str = None,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    delay: float = 0.5
) -> List[str]:
    """
    Modify multiple code snippets with the same configuration.
    
    Args:
        codes: List of original code strings
        config: Pattern configuration dict (same for all codes)
        api_key: DeepSeek API key
        model: Model name
        temperature: Sampling temperature
        delay: Delay between API calls (seconds) to avoid rate limiting
    
    Returns:
        List of modified code strings
    """
    modified_codes = []
    
    for i, code in enumerate(codes):
        print(f"Processing code {i + 1}/{len(codes)}...")
        
        try:
            modified = modify_code_with_deepseek(
                code=code,
                config=config,
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            modified_codes.append(modified)
        except Exception as e:
            print(f"Error modifying code {i + 1}: {e}")
            modified_codes.append(code)  # Fallback to original
        
        # Add delay to avoid rate limiting
        if i < len(codes) - 1:
            time.sleep(delay)
    
    return modified_codes


# ==================== Usage Examples ====================

if __name__ == "__main__":
    # Set your API key
    # os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"
    
    original_code = """
def last_occurence_char(string, char):
    flag = -1
    for i in range(len(string)):
        if string[i] == char:
            flag = i
    if flag == -1:
        return None
    else:
        return flag + 1
"""
    
    # ========== Example 1: 同时应用多个patterns（一次调用） ==========
    print("=" * 60)
    print("Example 1: 同时修改 docstrings + comments + type_hints")
    print("=" * 60)
    
    config = {
        'docstrings': 2,      # 会应用
        'comments': 2,        # 会应用
        'type_hints': 1,      # 会应用
        'error_handling': 0,  # 不应用
        'variable_style': 1,  # 会应用
    }
    
    # 一次API调用，同时应用4个patterns！
    result = modify_code_with_deepseek(original_code, config)
    
    print("\nOriginal:")
    print(original_code)
    print("\nModified (应用了 docstrings + comments + type_hints + variable_style):")
    print(result)
    
    # ========== Example 2: 只修改一个pattern ==========
    print("\n" + "=" * 60)
    print("Example 2: 只修改 docstrings")
    print("=" * 60)
    
    config_single = {
        'docstrings': 1,
        'comments': 0,
        'type_hints': 0,
        'error_handling': 0,
        'variable_style': 0,
    }
    
    result_single = modify_code_with_deepseek(original_code, config_single)
    print("\nModified:")
    print(result_single)
    
    # ========== Example 3: 全部patterns都修改（aggressive） ==========
    print("\n" + "=" * 60)
    print("Example 3: 修改全部5个patterns")
    print("=" * 60)
    
    config_all = {
        'docstrings': 3,
        'comments': 3,
        'type_hints': 3,
        'error_handling': 3,
        'variable_style': 3,
    }
    
    result_all = modify_code_with_deepseek(original_code, config_all)
    print("\nModified:")
    print(result_all)
    
    # ========== Example 4: Batch处理多个代码 ==========
    print("\n" + "=" * 60)
    print("Example 4: 批量处理3个代码（同样的config）")
    print("=" * 60)
    
    codes = [
        """
def add(a, b):
    return a + b
""",
        """
def multiply(x, y):
    result = x * y
    return result
""",
        """
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
"""
    ]
    
    batch_config = {
        'docstrings': 2,
        'type_hints': 1,
        'variable_style': 1,
    }
    
    batch_results = modify_codes_batch(codes, batch_config, delay=1.0)
    
    for i, (orig, modified) in enumerate(zip(codes, batch_results), 1):
        print(f"\n--- Code {i} ---")
        print("Original:")
        print(orig.strip())
        print("\nModified:")
        print(modified)