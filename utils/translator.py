"""
Simple Translation Manager
简单翻译管理器，支持命令行语言设置
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict


class SimpleTranslator:
    """Simple translation manager"""

    def __init__(self, locales_dir: str = "locales", default_language: str = "en"):
        self.locales_dir = Path(locales_dir)
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Any]] = {}
        self._load_translations()

    def _load_translations(self):
        """Load all language files"""
        if not self.locales_dir.exists():
            print(f"Warning: Locales directory {self.locales_dir} not found")
            return

        for lang_file in self.locales_dir.glob("*.json"):
            lang_code = lang_file.stem
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    self.translations[lang_code] = json.load(f)
                print(f"Loaded {lang_code} language translations")
            except Exception as e:
                print(f"Error loading {lang_file}: {e}")

    def set_language(self, language: str):
        """Set current language"""
        if language in self.translations:
            self.current_language = language
            print(f"Language set to: {language}")
        else:
            print(f"Warning: Language {language} not found, using default {self.default_language}")
            self.current_language = self.default_language

    def get_available_languages(self) -> Dict[str, str]:
        """Get available language list"""
        available_languages = {}

        # Scan locales folder for all json files
        for lang_file in self.locales_dir.glob("*.json"):
            lang_code = lang_file.stem

            # Language display name mapping
            display_names = {
                "en": "English",
                "zh": "中文",
                "ja": "日本語",
                "ko": "한국어",
                "fr": "Français",
                "de": "Deutsch",
                "es": "Español",
                "ru": "Русский",
                "it": "Italiano",
                "pt": "Português",
            }

            display_name = display_names.get(lang_code, lang_code.upper())
            available_languages[lang_code] = display_name

        return available_languages

    def t(self, key: str, **kwargs) -> str:
        """Translation function"""
        # Support nested keys like "table_headers.no"
        keys = key.split(".")

        # Get translation for current language
        translation = self.translations.get(self.current_language, {})

        # If current language has no translation, try default language
        if not translation:
            translation = self.translations.get(self.default_language, {})

        # Navigate key hierarchy
        try:
            for k in keys:
                translation = translation[k]
        except (KeyError, TypeError):
            # If translation not found, return key name
            return key

        # If string and needs formatting
        if isinstance(translation, str) and kwargs:
            try:
                return translation.format(**kwargs)
            except (KeyError, ValueError):
                return translation

        return translation

    def get_current_language(self) -> str:
        """Get current language"""
        return self.current_language

    def get_language_display_name(self) -> str:
        """Get current language display name"""
        available = self.get_available_languages()
        return available.get(self.current_language, self.current_language)


# Global translator instance
translator = SimpleTranslator()


def initialize_from_args():
    """Initialize from command line arguments only"""
    command_lang = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--lang="):
                lang_param = arg.split("=")[1]
                # Manual check for available languages
                if lang_param in ["en", "zh", "ja"]:  # Simplified check
                    command_lang = lang_param
                    break

    # Set language from command line
    if command_lang:
        translator.set_language(command_lang)
        print(f"Using command line language: {command_lang}")
    else:
        # Use default language
        print(f"No language specified, using default: {translator.default_language}")


# Global translation functions
def _(key: str, **kwargs) -> str:
    """Global translation function shorthand"""
    return translator.t(key, **kwargs)


def set_language(language: str):
    """Set language global function"""
    translator.set_language(language)


def get_language_display_name() -> str:
    """Get current language display name global function"""
    return translator.get_language_display_name()


def get_available_languages() -> Dict[str, str]:
    """Get available language list global function"""
    return translator.get_available_languages()


def get_current_language() -> str:
    """Get current language global function"""
    return translator.get_current_language()


# Manual initialization - do NOT auto-initialize on import
# initialize_from_args()


# Global initialization function to call from main.py
def initialize_from_main():
    """Initialize from main.py with command line support"""
    command_lang = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--lang="):
                lang_param = arg.split("=")[1]
                # Simplified available languages check
                if lang_param in ["en", "zh", "ja"]:
                    command_lang = lang_param
                    break

    # Set language from command line
    if command_lang:
        translator.set_language(command_lang)
        print(f"Using command line language: {command_lang}")
    else:
        # Use default language
        print(f"No language specified, using default: {translator.default_language}")

