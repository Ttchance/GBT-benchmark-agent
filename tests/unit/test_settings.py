import importlib
import os
import unittest
from unittest.mock import patch

import config.settings as settings


class SettingsEnvTest(unittest.TestCase):
    def tearDown(self):
        importlib.reload(settings)

    def test_llm_api_config_strips_surrounding_whitespace(self):
        env = {
            "OPENAI_API_KEY": " sk-proxy\n",
            "OPENAI_BASE_URL": " https://proxy.example/v1\r\n",
            "AZURE_OPENAI_API_KEY": "\tsk-azure\n",
            "AZURE_OPENAI_ENDPOINT": " https://azure.example/ \n",
            "MULTIMODAL_OPENAI_API_KEY": " sk-multimodal\r\n",
        }
        with patch.dict(os.environ, env, clear=False):
            reloaded = importlib.reload(settings)

        self.assertEqual(reloaded.LLM_CONFIG["api_key"], "sk-proxy")
        self.assertEqual(reloaded.LLM_CONFIG["base_url"], "https://proxy.example/v1")
        self.assertEqual(reloaded.AZURE_LLM_CONFIG["api_key"], "sk-azure")
        self.assertEqual(reloaded.AZURE_LLM_CONFIG["azure_endpoint"], "https://azure.example/")
        self.assertEqual(reloaded.MULTIMODAL_LLM_CONFIG["api_key"], "sk-multimodal")


if __name__ == "__main__":
    unittest.main()
