import os
import time
import logging
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _retry(on_exception=Exception, retries=2, delay=1.0):
    def deco(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except on_exception as e:
                    last_exc = e
                    logger.warning("Attempt %s failed: %s", attempt + 1, e)
                    time.sleep(delay * (attempt + 1))
            raise last_exc
        return wrapper
    return deco


class ChatbotService:
    """Robust Chatbot service wrapper around Hugging Face InferenceClient.

    - Loads config from environment but supports explicit token for testing.
    - Defensively trims conversation history and sanitizes product context.
    - Performs simple retry for transient errors.
    """

    DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", "katanemo/Arch-Router-1.5B")
    MAX_HISTORY = 12  # keep last N messages to avoid very long payloads
    PRODUCT_LIMIT = 30

    def __init__(self, token: Optional[str] = None):
        load_dotenv()
        self.token = token or os.getenv("HUGGINGFACE_API_KEY")
        if not self.token:
            # don't raise at import time — raise where used; create object but mark unavailable
            logger.error("HUGGINGFACE_API_KEY not found in environment")
            self.client = None
        else:
            self.client = InferenceClient(token=self.token)

        self.model = self.DEFAULT_MODEL

        # System instruction for the model. Keep it explicit and human-readable
        # so it is easy to edit later.
        self.system_instruction = (
            "You are a professional e-shop assistant. Your ONLY purpose is to help customers with:\n"
            "- Product information and search\n"
            "- Answering questions about available products\n"
            "- Recommendations based on customer needs\n"
            "- Help with orders and purchasing\n\n"
            "STRICT RULES:\n"
            "- ONLY answer questions related to the e-shop and its products.\n"
            "- DO NOT tell jokes, provide general knowledge, or discuss topics outside the shop.\n"
            "- ALWAYS redirect customers back to shop-related topics in a polite manner.\n"
        )

    # Removed off-topic detection and related keyword rules as requested.
    # The service no longer performs client-side off-topic filtering.

    def _sanitize_products(self, products: Optional[List[Dict]]) -> List[Dict]:
        if not products:
            return []
        sanitized = []
        for p in products[: self.PRODUCT_LIMIT]:
            try:
                sanitized.append({
                    'name': str(p.get('name', 'N/A')),
                    'description': str(p.get('description', '') or ''),
                    'price': float(p.get('price', 0) or 0),
                    'stock': int(p.get('stock', 0) or 0)
                })
            except Exception:
                # fallback, keep parsing resilient
                sanitized.append({'name': 'N/A', 'description': '', 'price': 0.0, 'stock': 0})
        return sanitized

    def _build_messages(self, user_message: str, chat_history: List[Dict], products: Optional[List[Dict]] = None) -> List[Dict]:
        # trim history to last MAX_HISTORY messages
        history = (chat_history or [])[-self.MAX_HISTORY:]

        system_message = self.system_instruction
        prod_list = self._sanitize_products(products)
        if prod_list:
            system_message += "\n\nAvailable products (first %s):\n" % len(prod_list)
            for p in prod_list:
                system_message += f"- {p['name']}: {p['description']} (Price: €{p['price']:.2f}, Stock: {p['stock']})\n"

        messages = [{'role': 'system', 'content': system_message}]
        messages.extend(history)
        messages.append({'role': 'user', 'content': user_message})
        return messages

    @_retry(on_exception=Exception, retries=2, delay=0.7)
    def _call_model(self, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.7) -> Dict:
        if not self.client:
            raise RuntimeError('Hugging Face client is not initialized; missing API token')

        # keep the same method name as previously used (chat_completion)
        response = self.client.chat_completion(messages=messages, model=self.model, max_tokens=max_tokens, temperature=temperature)
        return response

    def get_chatbot_response(self, user_message: str, chat_history: Optional[List[Dict]] = None) -> Dict:
        chat_history = chat_history or []

        try:
            messages = self._build_messages(user_message, chat_history)
            response = self._call_model(messages=messages)

            if response and getattr(response, 'choices', None) and len(response.choices) > 0:
                # compatible with HF response shape used previously
                response_text = response.choices[0].message.content.strip()
                return {'response': response_text, 'success': True}

            return {'response': "Sorry, I couldn't generate a response. Please try again.", 'success': False, 'error': 'No response from API'}

        except Exception as e:
            logger.exception('Error while calling Hugging Face API')
            return {'response': 'Sorry, an error occurred. Please try again later.', 'success': False, 'error': str(e)}

    def get_chatbot_response_with_products(self, user_message: str, chat_history: Optional[List[Dict]] = None, products: Optional[List[Dict]] = None) -> Dict:
        chat_history = chat_history or []

        try:
            messages = self._build_messages(user_message, chat_history, products)
            response = self._call_model(messages=messages)

            if response and getattr(response, 'choices', None) and len(response.choices) > 0:
                response_text = response.choices[0].message.content.strip()
                return {'response': response_text, 'success': True}

            return {'response': "Sorry, I couldn't generate a response. Please try again.", 'success': False, 'error': 'No response from API'}

        except Exception as e:
            logger.exception('Error while calling Hugging Face API with products')
            return {'response': 'Sorry, an error occurred. Please try again later.', 'success': False, 'error': str(e)}