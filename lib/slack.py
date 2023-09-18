"""
Slack Client Module

This module provides a SlackClient class for managing interactions with the Slack API. 
It allows for the retrieval of chat history, including main messages and threaded replies, 
from a specified Slack channel within a given time range. The module also provides utilities 
for formatting the retrieved messages, replacing user IDs with user names, and handling various 
message subtypes.

Classes:
    SlackClient: Manages interactions with the Slack API and provides methods for retrieving and formatting chat history.

Constants:
    SKIP_SUMMARY_TAG, ADD_SUMMARY_TAG, POST_SUMMARY_TAG: Tags for channel summary management.
    REPLY_PREFIX: Prefix added to threaded replies in the formatted chat history.
    EXCLUDED_SUBTYPES, CHANNEL_SUBTYPES, FILE_SUBTYPES, PIN_SUBTYPES, SYSTEM_SUBTYPES: Lists of message subtypes for handling.

Example:
    ```
    client = SlackClient(SLACK_BOT_TOKEN)
    start_time = datetime(2022, 5, 1, 0, 0, 0)
    end_time = datetime(2022, 5, 2, 0, 0, 0)
    messages = client.load_messages('C12345678', start_time, end_time)
    ```

Note:
    Requires the `slack_sdk` package.
"""

import re
import sys
import time
from datetime import datetime
from slack_sdk.errors import SlackApiError
from slack_sdk import WebClient
from lib.utils import retry, sort_by_numeric_prefix

# Constants
SKIP_SUMMARY_TAG = "#skip-summary"
ADD_SUMMARY_TAG = "#add-summary"
POST_SUMMARY_TAG = "#post-summary"
REPLY_PREFIX = " -> "
EXCLUDED_SUBTYPES = ["bot_message", "channel_archive", "me_message",
                     "message_change", "message_deleted", "message_replied",
                     "reminder_add"]
CHANNEL_SUBTYPES = ["channel_join", "channel_leave", "channel_topic",
                    "channel_purpose", "channel_name", "channel_unarchive"]
FILE_SUBTYPES = ["file_share", "file_comment", "file_mention"]
PIN_SUBTYPES = ["pinned_item", "unpinned_item"]
ORIGINAL_SUBTYPES = ["system"]
SYSTEM_SUBTYPES = CHANNEL_SUBTYPES + FILE_SUBTYPES + PIN_SUBTYPES + ORIGINAL_SUBTYPES

class SlackClient:
    """ 
    A class for managing interactions with the Slack API.

    The SlackClient class provides methods for retrieving chat history from a specified Slack channel 
    within a given time range. It handles main messages, threaded replies, and various message subtypes. 
    The class also offers utilities for formatting the retrieved messages, replacing user IDs with user names, 
    and managing API call rate limits.

    Args:
        slack_api_token (str): The Slack Bot token used to authenticate with the Slack API.
        debug_mode (bool, optional): If set to True, additional debug information will be printed. Defaults to False.

    Attributes:
        client (WebClient): The Slack WebClient object used for API calls.
        users (list): A list of dictionaries containing information about each user in the Slack workspace.
        channels (list): A list of dictionaries containing information about each public channel in the Slack workspace.
        debug_mode (bool): Indicates whether debug mode is active.

    Methods:
        post_message(text, channel): Post a message to a specified Slack channel.
        load_messages(channel_id, start_time, end_time): Retrieve and format the chat history for the specified channel between the given start and end times.
        get_user_name(user_id): Get the display name of a user with the given ID.
        replace_user_id_with_name(body_text): Replace user IDs in a chat message text with user display names.
    """

    def __init__(self, slack_api_token: str, debug_mode: bool = False):
        self.client = WebClient(token=slack_api_token)
        self.users = self._get_users_info()
        self.channels = self._get_channels_info()
        self.debug_mode = debug_mode

    def post_message(self, text: str, channel):
        """
        Post a message to a specified Slack channel.

        Args:
            text (str): The text of the message to be posted.
            channel (str): The ID of the channel to post the message to.

        Raises:
            SlackApiError: If an error occurs while attempting to post the message.

        Example:
            ```
            client = SlackClient(SLACK_BOT_TOKEN)
            client.post_message("Hello, World!", "YOUR_CHANNEL_ID")
            ```
        """
        response = self.client.chat_postMessage(channel=channel, text=text)
        if not response["ok"]:
            print(f'Failed to post message: {response["error"]}')
            raise SlackApiError('Failed to post message', response["error"])

    def load_messages(self, channel_id: str, start_time: datetime,
                    end_time: datetime) -> list:
        """ 
        Retrieve and format the chat history for the specified channel between the given start and end times.

        This method fetches the chat history, including main messages and threaded replies, from the specified Slack channel 
        within the provided time range. It handles various message subtypes and provides utilities for formatting the 
        retrieved messages. If a threaded message's parent is not present in the retrieved messages, a placeholder message 
        "System: Earlier message not retrieved" is added.

        Args:
            channel_id (str): The ID of the channel to retrieve the chat history for.
            start_time (datetime): The start time of the time range to retrieve chat history for.
            end_time (datetime): The end time of the time range to retrieve chat history for.

        Returns:
            list: A list of formatted chat messages from the specified channel. The list includes both main messages and 
                threaded replies. Returns None if there are no messages or an error occurs.

        Raises:
            SlackApiError: If an error occurs while attempting to retrieve the chat history.

        Examples:
            >>> start_time = datetime(2022, 5, 1, 0, 0, 0)
            >>> end_time = datetime(2022, 5, 2, 0, 0, 0)
            >>> messages = client.load_messages('C12345678', start_time, end_time)
            >>> for msg in messages:
            ...     print(msg)
            ...
            Alice: Hi, Bob! How's it going?
            -> Bob: It's going well, Alice! Thanks for asking.
            -> Charlie: Hey Bob, did you finish that report?
            Dave: Good morning everyone!
            System: Earlier message not retrieved
            -> Eve: I had a question about that too.
        """

        @retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
        def _join_conversations(channel):
            return self.client.conversations_join(channel=channel)

        @retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
        def _fetch_conversations_history(channel, oldest, latest, limit, cursor=None):
            return self.client.conversations_history(
                channel=channel,
                oldest=oldest,
                latest=latest,
                limit=limit,
                cursor=cursor)

        @retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
        def _fetch_conversations_replies(channel, timestamp,oldest, latest, limit, cursor=None):
            return self.client.conversations_replies(
                channel=channel,
                ts=timestamp,
                oldest=oldest,
                latest=latest,
                limit=limit,
                cursor=cursor)

        messages_info = []
        next_cursor = None  # 初期のカーソルをNoneに設定

        while True:
            self._wait_api_call()
            try:
                result = _fetch_conversations_history(
                    channel=channel_id,
                    oldest=str(start_time.timestamp()),
                    latest=str(end_time.timestamp()),
                    limit=1000,
                    cursor=next_cursor)  # カーソルを使用してメッセージを取得
            except SlackApiError as error:
                if error.response['error'] == 'not_in_channel':
                    self._wait_api_call()
                    response = _join_conversations(channel=channel_id)
                    if not response["ok"]:
                        print("Error: Failed conversations_join()")
                        sys.exit(1)
                    continue  # チャンネルに参加した後、再度メッセージの取得を試みる
                print(f"Error: {error}")
                return None

            if result is None:
                print("Error: Result is None")
                return None
 
            messages_info.extend(result["messages"])

            if result["has_more"]:
                next_cursor = result['response_metadata']['next_cursor']
            else:
                break  # すべてのメッセージを取得した場合、ループを終了

        if self.debug_mode:
            print(f"Total messages fetched: {len(messages_info)}")
            for debug_msg in messages_info:
                print(debug_msg)

        # Filter out messages with EXCLUDED_SUBTYPES and bot_id
        messages = list(filter(lambda m: m.get("subtype") not in EXCLUDED_SUBTYPES
                               and "bot_id" not in m, messages_info))

        # Mark messages for fetching replies and filter out messages that don't require text
        filtered_messages = []
        added_thread_starts = set()  # To track which thread starts have been added

        for message in messages:
            if "thread_ts" in message:
                if message["ts"] == message["thread_ts"]:
                    # Mark thread start messages
                    message["fetch_replies"] = True
                    filtered_messages.append(message)
                else:
                    # Check if the thread's start message is already in the messages list
                    if any(msg["ts"] == message["thread_ts"] for msg in messages):
                        continue  # Skip this message as it doesn't require text
                    # If the thread's start message is not in the list, add a dummy start message
                    if message["thread_ts"] not in added_thread_starts:
                        dummy_thread_start = {
                            "type": "message",
                            "subtype": "system",
                            "text": "System: Earlier message not retrieved",
                            "ts": message["thread_ts"],
                            "fetch_replies": True,
                            "user": "System"
                        }
                        filtered_messages.append(dummy_thread_start)
                        added_thread_starts.add(message["thread_ts"])
                    message["fetch_replies"] = True
                    filtered_messages.append(message)
            else:
                message["fetch_replies"] = False
                filtered_messages.append(message)

        messages = filtered_messages

        if len(messages) < 1:
            return None

        all_target_messages = []
        for message in messages:
            all_target_messages.append(message)  # Add the original message to the new list

            if message.get("fetch_replies", True):
                next_cursor = None
                while True:
                    self._wait_api_call()
                    try:
                        thread_replies = _fetch_conversations_replies(
                            channel=channel_id,
                            timestamp=message["ts"],
                            oldest=str(start_time.timestamp()),
                            latest=str(end_time.timestamp()),
                            limit=1000,
                            cursor=next_cursor)
                    except SlackApiError as error:
                        print(f"Error: Fetching thread replies: {error}")
                        break

                    if thread_replies is None:
                        print("Error: Thread replies result is None")
                        break

                    # Exclude the parent message as it's already in the messages list
                    for reply in thread_replies["messages"][1:]:
                        all_target_messages.append(reply)  # Add each reply to the new list

                    if thread_replies["has_more"]:
                        next_cursor = thread_replies['response_metadata']['next_cursor']
                    else:
                        break  # All replies fetched, exit the loop

        messages = all_target_messages

        messages_texts = []
        for message in messages[::-1]:
            # Ignore empty messages
            if len(message["text"].strip()) == 0:
                continue

            # Determine the speaker based on the subtype
            if "subtype" in message and message["subtype"] in SYSTEM_SUBTYPES:
                speaker_name = "System"
            else:
                # Get speaker name
                speaker_name = self.get_user_name(message["user"]) or "Somebody"

            # Get message body from result dict.
            body_text = message["text"].replace("\n", "\\n")

            # Replace User IDs in a chat message text with user names.
            body_text = self.replace_user_id_with_name(body_text)

            # Replace all channel ids with "other channel"
            body_text = re.sub(r"<#[A-Z0-9]+>", " other channel ", body_text)

            # Construct the final message format
            body_text = f"{speaker_name}: {body_text}"

            # Determine if the message is a reply
            if "thread_ts" in message and message["ts"] != message["thread_ts"]:
                messages_texts.append(REPLY_PREFIX + body_text)
            else:
                messages_texts.append(body_text)

        if self.debug_mode:
            for debug_msg in messages_texts:
                print(debug_msg)

        if len(messages_texts) == 0:
            return None
        return messages_texts

    def get_user_name(self, user_id: str) -> str:
        """ Get the name of a user with the given ID.

        Args:
            user_id (str): The ID of the user to look up.

        Returns:
            str: The name of the user with the given ID, or None if no such user exists.

        Examples:
            >>> users = [{'id': 'U1234', 'name': 'Alice'}, {'id': 'U5678', 'name': 'Bob'}]
            >>> get_user_name('U1234', users)
            'Alice'
            >>> get_user_name('U9999', users)
            None
        """
        matching_users = [user for user in self.users if user['id'] == user_id]
        return matching_users[0]['profile']['display_name'] if len(matching_users) > 0 else None

    def replace_user_id_with_name(self, body_text: str) -> str:
        """ Replace user IDs in a chat message text with user names.

        Args:
            body_text (str): The text of a chat message.
            users (list): A list of user information dictionaries.
                Each dictionary must have 'id' and 'name' keys.

        Returns:
            str: The text of the chat message with user IDs replaced with user names.

        Examples:
            >>> users = [{'id': 'U1234', 'name': 'Alice'}, {'id': 'U5678', 'name': 'Bob'}]
            >>> body_text = "Hi <@U1234>, how are you?"
            >>> replace_user_id_with_name(body_text, users)
            "Hi @Alice, how are you?"
        """
        pattern = r"<@([A-Z0-9]+)>"
        for match in re.finditer(pattern, body_text):
            user_id = match.group(1)
            user_name = next(
                (user['name'] for user in self.users if user['id'] == user_id),
                user_id)
            body_text = body_text.replace(match.group(0), user_name)
        return body_text

    def _get_users_info(self, wait_time=3) -> list:
        """
        Retrieve information about all users in the Slack workspace.

        This method retrieves a list of dictionaries containing information about each user,
        including their ID, name, and other metadata. It handles pagination by making multiple
        API calls if necessary.

        Args:
            wait_time (int, optional): The time to wait between API calls in seconds. Defaults to 3 seconds.

        Returns:
            list: A list of dictionaries containing information about each user,
                including their ID, name, and other metadata.

        Raises:
            SlackApiError: If an error occurs while attempting to retrieve the user information.

        Examples:
            >>> users = get_users_info(wait_time=5)
            >>> print(users[0])
            {
                'id': 'U12345678',
                'name': 'alice',
                'real_name': 'Alice Smith',
                'email': 'alice@example.com',
                ...
            }
        """
        @retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
        def _fetch_users_info(cursor=None, limit=100):
            return self.client.users_list(cursor=cursor, limit=limit)

        try:
            users = []
            next_cursor = None
            while True:
                self._wait_api_call()
                users_info = _fetch_users_info(cursor=next_cursor)
                time.sleep(wait_time)
                users.extend(users_info['members'])
                if users_info["response_metadata"]["next_cursor"]:
                    next_cursor = users_info["response_metadata"]["next_cursor"]
                else:
                    break
            return users
        except SlackApiError as error:
            print(f"Error: {error}")
            sys.exit(1)

    def _get_channels_info(self) -> list:
        """
        Retrieve information about all public channels in the Slack workspace.

        Returns:
            list: A list of dictionaries containing information about each public channel, including its ID, name, and other metadata. sorted by channel name.

        Raises:
            SlackApiError: If an error occurs while attempting to retrieve the channel information.

        Examples:
            >>> channels = get_channels_info()
            >>> print(channels[0])
            {
                'id': 'C12345678',
                'name': 'general',
                'is_channel': True,
                'is_archived': False,
                ...
            }
        """ 
        @retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)
        def _fetch_channels_info():
            return self.client.conversations_list(
                    types="public_channel", exclude_archived=True, limit=1000)

        try:
            self._wait_api_call()
            result = _fetch_channels_info()
            channels_info = [
                channel for channel in result['channels']
                if not channel["is_archived"] and channel["is_channel"]
                    and ((not channel["is_ext_shared"] and not channel["is_org_shared"])
                          or ADD_SUMMARY_TAG in channel["purpose"]["value"])
                    and SKIP_SUMMARY_TAG not in channel["purpose"]["value"]
            ]
            channels_info = sort_by_numeric_prefix(channels_info,
                                                    get_key=lambda x: x["name"])
            return channels_info
        except SlackApiError as error:
            print(f"Error: {error}")
            sys.exit(1)

    def _wait_api_call(self):
        """ most of api call limit is 20 per minute """
        time.sleep(60 / 20)
