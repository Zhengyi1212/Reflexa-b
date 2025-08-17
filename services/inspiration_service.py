# services/inspiration_service.py
import json
import random
from typing import List, Dict, Optional

class InspirationService:
    """
    一个轻量级的服务，用于管理和提供预设的p5.js灵感代码。
    它取代了之前复杂的RAG服务，专注于从一个JSON文件中进行查找。
    """
    def __init__(self):
        """
        初始化灵感服务。
        """
        print("Initializing Inspiration Service...")
        self._examples: List[Dict] = []
        self._tag_to_code: Dict[str, str] = {}
        print("✅ Inspiration Service initialized.")

    def load_examples(self, filepath: str):
        """
        从指定的JSON文件加载代码示例。
        这个函数应该在应用启动时被调用一次。
        """
        print(f"Loading inspiration examples from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self._examples = json.load(f)
            
            # 创建一个从标签到代码的快速查找字典
            self._tag_to_code = {example['tag']: example['code'] for example in self._examples}
            
            print(f"✅ Successfully loaded {len(self._examples)} inspiration examples.")
        except FileNotFoundError:
            print(f"❌ Error: Inspiration data file not found at {filepath}")
        except json.JSONDecodeError:
            print(f"❌ Error: Failed to decode JSON from {filepath}")
        except Exception as e:
            print(f"❌ An unexpected error occurred while loading examples: {e}")

    def get_random_styles(self, count: int = 3) -> List[Dict[str, str]]:
        """
        ‼️【新方法，替换 get_random_tags】
        从加载的示例中随机选择指定数量的风格。
        返回一个字典列表，每个字典包含 'tag' 和 'image'。
        """
        if not self._examples:
            print("⚠️ No examples loaded, cannot provide random styles.")
            return []
        
        # 确保示例包含 'tag' 和 'image' 键，避免运行时错误
        valid_examples = [
            ex for ex in self._examples if 'tag' in ex and 'image' in ex
        ]

        if len(valid_examples) <= count:
            # 如果可用示例不足，返回所有有效示例
            print(f"⚠️ Not enough valid examples to provide {count} unique styles. Returning all {len(valid_examples)}.")
            return [{'tag': ex['tag'], 'image': ex['image']} for ex in valid_examples]
            
        # 随机选择不重复的示例
        random_samples = random.sample(valid_examples, count)
        
        # 提取 tag 和 image 字段
        result = [{'tag': sample['tag'], 'image': sample['image']} for sample in random_samples]
        print(f"Provided random styles: {result}")
        return result

    def get_code_by_tag(self, tag: str) -> Optional[str]:
        """
        根据标签查找并返回对应的代码。(此方法保持不变)
        """
        code = self._tag_to_code.get(tag)
        if code:
            print(f"Found code for tag: '{tag}'")
        else:
            print(f"⚠️ Could not find code for tag: '{tag}'")
        return code
