# fractal_framework/utils/language_processors.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import ast
import re
from dataclasses import dataclass
from enum import Enum

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JULIA = "julia"
    RUST = "rust"
    GO = "go"
    CPP = "cpp"
    JAVA = "java"
    # Add more languages as needed

@dataclass
class CodeAnalysis:
    """Analysis results for a code snippet"""
    language: LanguageType
    imports: List[str]
    functions: List[str]
    classes: List[str]
    variables: List[str]
    patterns: List[str]
    tokens: List[str]
    metadata: Dict[str, Any]

class LanguageProcessor(ABC):
    """Base class for language-specific processors"""

    @abstractmethod
    def detect_language(self, code: str) -> bool:
        """Check if this processor can handle the code"""
        pass

    @abstractmethod
    def preprocess(self, code: str) -> str:
        """Preprocess code for vectorization"""
        pass

    @abstractmethod
    def analyze(self, code: str) -> CodeAnalysis:
        """Analyze code structure and patterns"""
        pass

    @abstractmethod
    def tokenize(self, code: str) -> List[str]:
        """Tokenize code for vectorization"""
        pass

    @abstractmethod
    def extract_documentation(self, code: str) -> str:
        """Extract documentation and comments"""
        pass

class PythonProcessor(LanguageProcessor):
    """Python code processor"""

    def detect_language(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except:
            return False

    def preprocess(self, code: str) -> str:
        # Remove comments and docstrings
        try:
            tree = ast.parse(code)
            return ast.unparse(tree)
        except:
            return code

    def analyze(self, code: str) -> CodeAnalysis:
        imports = []
        functions = []
        classes = []
        variables = []
        patterns = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(n.name for n in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(f"{node.module}")
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    variables.append(node.id)

            # Detect patterns
            patterns = self._detect_patterns(tree)

            return CodeAnalysis(
                language=LanguageType.PYTHON,
                imports=imports,
                functions=functions,
                classes=classes,
                variables=variables,
                patterns=patterns,
                tokens=self.tokenize(code),
                metadata={}
            )
        except:
            return CodeAnalysis(
                language=LanguageType.PYTHON,
                imports=[], functions=[], classes=[],
                variables=[], patterns=[], tokens=[],
                metadata={}
            )

    def _detect_patterns(self, tree: ast.AST) -> List[str]:
        patterns = []
        # Add pattern detection logic here
        return patterns

    def tokenize(self, code: str) -> List[str]:
        tokens = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    tokens.append(node.id)
                elif isinstance(node, ast.Str):
                    tokens.append(node.s)
        except:
            # Fallback to simple tokenization
            tokens = re.findall(r'\w+', code)
        return tokens

    def extract_documentation(self, code: str) -> str:
        docs = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docs.append(docstring)
        except:
            pass
        return "\n".join(docs)

class JuliaProcessor(LanguageProcessor):
    """Julia code processor"""

    def detect_language(self, code: str) -> bool:
        # Check for Julia-specific syntax
        julia_patterns = [
            r'function\s+\w+',
            r'end\s*$',
            r'module\s+\w+',
            r'using\s+\w+',
            r'::\s*\w+'
        ]
        return any(re.search(pattern, code, re.MULTILINE) for pattern in julia_patterns)

    def preprocess(self, code: str) -> str:
        # Remove comments
        code = re.sub(r'#=.*?=#', '', code, flags=re.DOTALL)  # Multiline comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Single line comments
        return code

    def analyze(self, code: str) -> CodeAnalysis:
        imports = re.findall(r'using\s+(\w+)', code)
        functions = re.findall(r'function\s+(\w+)', code)
        structs = re.findall(r'struct\s+(\w+)', code)
        variables = re.findall(r'(\w+)\s*=(?!=)', code)

        # Detect Julia-specific patterns
        patterns = []
        if re.search(r'@\w+', code):
            patterns.append('Uses macros')
        if re.search(r'::\s*\w+', code):
            patterns.append('Type annotations')
        if re.search(r'abstract\s+type', code):
            patterns.append('Abstract types')

        return CodeAnalysis(
            language=LanguageType.JULIA,
            imports=imports,
            functions=functions,
            classes=structs,
            variables=variables,
            patterns=patterns,
            tokens=self.tokenize(code),
            metadata={}
        )

    def tokenize(self, code: str) -> List[str]:
        # Basic tokenization for Julia
        return re.findall(r'\w+', self.preprocess(code))

    def extract_documentation(self, code: str) -> str:
        # Extract Julia docstrings
        docs = re.findall(r'"""(.*?)"""', code, re.DOTALL)
        return "\n".join(docs)

class LanguageProcessorRegistry:
    """Registry of language processors"""

    def __init__(self):
        self.processors: List[LanguageProcessor] = [
            PythonProcessor(),
            JuliaProcessor(),
            # Add more processors here
        ]

    def get_processor(self, code: str) -> Optional[LanguageProcessor]:
        """Get appropriate processor for the code"""
        for processor in self.processors:
            if processor.detect_language(code):
                return processor
        return None

    def add_processor(self, processor: LanguageProcessor):
        """Add a new language processor"""
        self.processors.append(processor)

    def analyze_code(self, code: str) -> Optional[CodeAnalysis]:
        """Analyze code using appropriate processor"""
        processor = self.get_processor(code)
        if processor:
            return processor.analyze(code)
        return None

    def preprocess_code(self, code: str) -> str:
        """Preprocess code using appropriate processor"""
        processor = self.get_processor(code)
        if processor:
            return processor.preprocess(code)
        return code
