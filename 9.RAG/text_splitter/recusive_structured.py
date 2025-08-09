from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = text = """
# random_example.py
import random
import string
import statistics
from typing import List, Tuple

class RandomDataGenerator:
    Generate random integers and random strings

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.seed = seed

    def random_int_list(self, n: int, lo: int = 0, hi: int = 100) -> List[int]:
      Return a list of n random integers between lo and hi (inclusive)
        return [random.randint(lo, hi) for _ in range(n)]

    def random_string(self, length: int = 8) -> str:
    Return a random alphanumeric string of given length.
        alphabet = string.ascii_letters + string.digits
        return ''.join(random.choice(alphabet) for _ in range(length))

def summarize_numbers(numbers: List[int]) -> Tuple[int, float, float]:

    Return (count, mean, stdev) for the provided list of numbers.
    If list has fewer than 2 items, stdev will be 0.0.

    if not numbers:
        return 0, 0.0, 0.0
    count = len(numbers)
    mean = statistics.mean(numbers)
    stdev = statistics.pstdev(numbers) if count < 2 else statistics.stdev(numbers)
    return count, mean, stdev

if __name__ == "__main__":
    rdg = RandomDataGenerator(seed=42)
    nums = rdg.random_int_list(10, lo=10, hi=50)
    rand_name = rdg.random_string(12)

    count, mean, stdev = summarize_numbers(nums)

    print("Random integers:", nums)
    print(f"Count: {count}, Mean: {mean:.2f}, Stdev: {stdev:.2f}")
    print("Random string:", rand_name)


"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(len(result))
print(result)