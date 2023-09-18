#!/usr/bin/env python3
"""
https://github.com/masuidrive/slack-summarizer
  by [masuidrive](https://twitter.com/masuidrive) @ [Bloom&Co., Inc.](https://www.bloom-and-co.com/)
  2023- [APACHE LICENSE, 2.0](https://www.apache.org/licenses/LICENSE-2.0)
"""
import os
import sys
from datetime import datetime, timedelta
import time
import pytz
import openai
import tiktoken
from slack_sdk.errors import SlackApiError
from lib.slack import SlackClient
from lib.slack import POST_SUMMARY_TAG
from lib.utils import remove_emoji, retry

# Load settings from environment variables
OPEN_AI_TOKEN = os.environ.get('OPEN_AI_TOKEN', '').strip()
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN', '').strip()
CHANNEL_ID = os.environ.get('SLACK_POST_CHANNEL_ID', '').strip()
LANGUAGE = str(os.environ.get('LANGUAGE') or "Japanese").strip()
TIMEZONE_STR = str(os.environ.get('TIMEZONE') or 'Asia/Tokyo').strip()
TEMPERATURE = float(os.environ.get('TEMPERATURE') or 0.3)
CHAT_MODEL = str(os.environ.get('CHAT_MODEL') or "gpt-3.5-turbo").strip()
ENCODING_MODEL = str(os.environ.get('ENCODING_MODEL') or "cl100k_base").strip()
DEBUG = str(os.environ.get('DEBUG') or "").strip() != ""
MAX_BODY_TOKENS = int(os.environ.get('MAX_BODY_TOKENS') or 3000)
REQUEST_INTERVAL = float(os.environ.get('REQUEST_INTERVAL') or 1/60)
SUMMARIZE_PROMPT = os.environ.get('SUMMARIZE_PROMPT', '').strip()
OUTPUT_SLACK = os.environ.get('OUTPUT_SLACK', '').strip()

def summarize(text: str, prompt_text: str, language: str, max_retries: int = 3, initial_wait_time: int = 2) -> str:
    """
    Summarize a chat log in bullet points, in the specified language, using a given prompt.

    Args:
        text (str): The chat log to summarize, in the format "Speaker: Message" separated by line breaks.
        prompt_text (str): The prompt to guide the summarization.
        language (str): The language to use for the summary.
        max_retries (int, optional): The maximum number of retries if the API call fails. Defaults to 3.
        initial_wait_time (int, optional): The initial wait time in seconds before the first retry. Defaults to 2.

    Returns:
        str: The summarized chat log in bullet point format.

    Examples:
        >>> summarize("Alice: Hi\nBob: Hello\nAlice: How are you?\nBob: I'm doing well, thanks.", "Summarize the following chat log.")
        '- Alice greeted Bob.\n- Bob responded with a greeting.\n- Alice asked how Bob was doing.\n- Bob replied that he was doing well.'
    """
    error_message = ""
    wait_time = initial_wait_time
    for i in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                temperature=TEMPERATURE,
                messages=[{
                    "role":
                    "system",
                    "content":
                    "\n".join([
                        'The chat log format consists of one line per message in the format "Speaker: Message".',
                        "The `\\n` within the message represents a line break."
                        f'The user understands {language} only.',
                        f'So, The assistant need to speak in {language}.',
                    ])
                }, {
                    "role":
                    "user",
                    "content":
                    "\n".join([
                        prompt_text,
                        f"Write in {language}.", "", text
                    ])
                }])
            time.sleep(REQUEST_INTERVAL)  # wait to avoid exceeding rate limit
            break
        except openai.error.ServiceUnavailableError as error:
            if DEBUG:
                print(f"Openai error: {error}")

            if i < max_retries - 1:  # i is zero indexed
                time.sleep(wait_time)  # wait before trying again
                wait_time *= 2  # double the wait time for the next retry
                continue
            else:
                error_message = "The service is currently unavailable. Please try again later."
                break
        except openai.error.Timeout as error:
            if DEBUG:
                print(f"Openai error: {error}")

            estimated_tokens = estimate_openai_chat_token_count(text)
            error_message = f"Timeout error occurred. The estimated token count is {estimated_tokens}. Please try again with shorter text."
            break
        except openai.error.APIConnectionError as error:
            if DEBUG:
                print(f"Openai error: {error}")
                
            if i < max_retries - 1:  # i is zero indexed
                time.sleep(wait_time)  # wait before trying again
                wait_time *= 2  # double the wait time for the next retry
                continue
            else:
                error_message = "A connection error occurred. Please check your internet connection and try again."
                break
        except openai.error.RateLimitError as error:
            if DEBUG:
                print(f"Openai error: {error}")

            if i < max_retries - 1:
                time.sleep(wait_time)
                wait_time *= 2
                continue
            else:
                error_message = "Exceeded rate limit. Please try again later."
                break

    if error_message:
        if DEBUG:
            print(f"Response: {error_message}")
        return error_message
        
    if DEBUG:
        print(f"Response:\n{response['choices'][0]['message']['content']}")

    return response["choices"][0]["message"]['content']

def get_time_range():
    """
    Get a time range starting from 25 hours ago and ending at the current time.

    Returns:
        tuple: A tuple containing the start and end times of the time range, as datetime objects.

    Examples:
        >>> start_time, end_time = get_time_range()
        >>> print(start_time, end_time)
        2022-05-17 09:00:00+09:00 2022-05-18 10:00:00+09:00
    """
    hours_back = 25
    timezone = pytz.timezone(TIMEZONE_STR)
    now = datetime.now(timezone)
    yesterday = now - timedelta(hours=hours_back)
    start_time = datetime(yesterday.year, yesterday.month, yesterday.day,
                          yesterday.hour, yesterday.minute, yesterday.second)
    end_time = datetime(now.year, now.month, now.day, now.hour, now.minute,
                        now.second)
    return start_time, end_time


def estimate_openai_chat_token_count(text: str) -> int:
    """
    Estimate the number of OpenAI API tokens that would be consumed by sending the given text to the chat API.

    Args:
        text (str): The text to be sent to the OpenAI chat API.

    Returns:
        int: The estimated number of tokens that would be consumed by sending the given text to the OpenAI chat API.

    Examples:
        >>> estimate_openai_chat_token_count("Hello, how are you?")
        7
    """
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    token_count = len(encoding.encode(text))

    return token_count

def split_messages_by_token_count(messages: list[str]) -> list[list[str]]:
    """
    Split a list of strings into sublists with a maximum token count.

    Args:
        messages (list[str]): A list of strings to be split.

    Returns:
        list[list[str]]: A list of sublists, where each sublist has a token count less than or equal to max_body_tokens.
    """
    body_token_counts = [
        estimate_openai_chat_token_count(message) for message in messages
    ]
    result = []
    current_sublist = []
    current_count = 0

    for message, count in zip(messages, body_token_counts):
        if current_count + count <= MAX_BODY_TOKENS:
            current_sublist.append(message)
            current_count += count
        else:
            result.append(current_sublist)
            current_sublist = [message]
            current_count = count

    result.append(current_sublist)
    return result


@retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
def post_summary(slack_client, summary, channel_id=None):
    """
    Post a summary to a Slack channel.

    Args:
        slack_client (SlackClient): The Slack client.
        summary (str): The summary to post.
        channel_id (str, optional): The ID of the channel to post to. If not provided, defaults to CHANNEL_ID.

    Raises:
        SlackApiError: If an error occurs.
    """
    if channel_id is None:
        channel_id = CHANNEL_ID
    slack_client.post_message(summary, channel_id)
    
def runner():
    """
    The main function to run the Slack summarizer application.
    
    This function performs the following steps:
    1. Validates the required environment variables.
    2. Sets up the OpenAI API key and Slack client.
    3. Determines the time range for message retrieval.
    4. Checks and formats the SUMMARIZE_PROMPT based on the presence of the {language} placeholder.
    5. Retrieves messages from Slack channels, summarizes them, and posts the summaries back to Slack.
    
    Raises:
        SystemExit: If any of the required environment variables are not set.
    """
    if OPEN_AI_TOKEN == "" or SLACK_BOT_TOKEN == "" or CHANNEL_ID == "" or SUMMARIZE_PROMPT == "":
        print("Error: OPEN_AI_TOKEN, SLACK_BOT_TOKEN, CHANNEL_ID, SUMMARIZE_PROMPT must be set.")
        sys.exit(1)
    
    # Set OpenAI API key
    openai.api_key = OPEN_AI_TOKEN

    # Set Slack Client
    slack_client = SlackClient(slack_api_token=SLACK_BOT_TOKEN, debug_mode=DEBUG)
    start_time, end_time = get_time_range()

    # Set Prompt Text
    if "{language}" in SUMMARIZE_PROMPT:
        prompt_text = SUMMARIZE_PROMPT.format(language=LANGUAGE)
    else:
        prompt_text = SUMMARIZE_PROMPT

    result_text = []
    for channel in slack_client.channels:
        messages = slack_client.load_messages(channel["id"], start_time,
                                              end_time)
        if DEBUG:
            print(f"Channel: {channel['name']}, {channel['id']}")
            print(f"Messages: \n{messages}")

        if messages is None:
            continue

        # remove emojis in messages
        messages = list(map(remove_emoji, messages))

        channel_summary = []
        for splitted_messages in split_messages_by_token_count(messages):
            text = summarize("\n".join(splitted_messages), prompt_text, LANGUAGE)
            channel_summary.append(text)

        # Post summary to the channel if #post-summary tag is in the channel description
        if POST_SUMMARY_TAG in channel["purpose"]["value"]:
            title = f"{start_time.strftime('%Y-%m-%d')} {channel['name']} summary\n\n"
            summary = title + "\n".join(channel_summary)
            post_summary(slack_client, summary, channel["id"])

        result_text.append(f"----\n<#{channel['id']}>\n")
        result_text.extend(channel_summary)

    title = (f"{start_time.strftime('%Y-%m-%d')} public channels summary\n\n")
    summary = title + "\n".join(result_text)

    if OUTPUT_SLACK:
        post_summary(slack_client, summary, CHANNEL_ID)

    if DEBUG:
        print(f"Summary: {summary}")

if __name__ == '__main__':
    runner()
