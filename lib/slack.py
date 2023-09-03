"""
Slack Client Module

This module provides a SlackClient class for managing a Slack bot client. 
It includes methods for posting messages, loading chat history, 
retrieving user and channel information, and replacing user IDs with names in messages.

Classes:
    SlackClient: Manages the Slack bot client.

Constants:
    SKIP_SUMMARY_TAG, ADD_SUMMARY_TAG, POST_SUMMARY_TAG: Tags for channel summary management.

Example:
    ```
    client = SlackClient(SLACK_BOT_TOKEN, SUMMARY_CHANNEL_ID)
    client.post("Hello, World!")
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

class SlackClient:
    """ 
    A class for managing a Slack bot client.

    This class provides methods to post messages to a Slack channel, load messages from a Slack channel,
    get user names, replace user IDs with names in a message, and retrieve information about all users 
    and public channels in the Slack workspace.

    Args:
        slack_api_token (str): The Slack Bot token used to authenticate with the Slack API.
        summary_channel (str): The Slack channel ID where the summary is posted.

    Attributes:
        client (WebClient): The Slack WebClient object used for API calls.
        users (list): A list of dictionaries containing information about each user in the Slack workspace.
        channels (list): A list of dictionaries containing information about each public channel in the Slack workspace.
        _summary_channel (str): The Slack channel ID where the summary is posted.

    Example:
        ```
        client = SlackClient(SLACK_BOT_TOKEN, SUMMARY_CHANNEL_ID)
        client.post("Hello, World!")
        ```

    Methods:
        post(text, channel=None): Post a message to a specified Slack channel.
        load_messages(channel_id, start_time, end_time): Load the chat history for the specified channel between the given start and end times.
        get_user_name(user_id): Get the name of a user with the given ID.
        replace_user_id_with_name(body_text): Replace user IDs in a chat message text with user names.
    """

    def __init__(self, slack_api_token: str, summary_channel: str):
        self.client = WebClient(token=slack_api_token)
        self.users = self._get_users_info()
        self.channels = self._get_channels_info()
        self._summary_channel = summary_channel
    
    def post(self, text: str, channel=None):
        """
        Post a message to a specified Slack channel.

        Args:
            text (str): The text of the message to be posted.
            channel (str, optional): The ID of the channel to post the message to. 
                Defaults to the summary channel specified during the initialization of the SlackClient object.

        Raises:
            SlackApiError: If an error occurs while attempting to post the message.

        Example:
            ```
            client = SlackClient(SLACK_BOT_TOKEN, SUMMARY_CHANNEL_ID)
            client.post("Hello, World!")
            ```
        """
        if channel is None:
            channel = self._summary_channel
        response = self.client.chat_postMessage(channel=channel, text=text)
        if not response["ok"]:
            print(f'Failed to post message: {response["error"]}')
            raise SlackApiError('Failed to post message', response["error"])
    
    def load_messages(self, channel_id: str, start_time: datetime,
                      end_time: datetime, wait_time: int = 5) -> list:
        """ 
        Load the chat history for the specified channel between the given start and end times.

        Args:
            channel_id (str): The ID of the channel to retrieve the chat history for.
            start_time (datetime): The start time of the time range to retrieve chat history for.
            end_time (datetime): The end time of the time range to retrieve chat history for.
            wait_time (int, optional): The time to wait before retrying the API call if an error occurs. Defaults to 5 seconds.

        Returns:
            list: A list of chat messages from the specified channel, in the format "Speaker: Message".
                Returns None if there are no messages or an error occurs.

        Raises:
            SlackApiError: If an error occurs while attempting to retrieve the chat history.

        Examples:
            >>> start_time = datetime(2022, 5, 1, 0, 0, 0)
            >>> end_time = datetime(2022, 5, 2, 0, 0, 0)
            >>> wait_time = 10
            >>> messages = client.load_messages('C12345678', start_time, end_time, wait_time)
            >>> print(messages[0])
            "Alice: Hi, Bob! How's it going?"
        """

        def _join_conversations(channel):
            return self.client.conversations_join(channel=channel)
        
        def _fetch_conversations_history(channel, oldest, latest, limit):
            return self.client.conversations_history(
                channel=channel,
                oldest=oldest,
                latest=latest,
                limit=limit)
        
        next_cursor = None
        messages_info = []
        fetch_conversations_history_retry = retry(max_retries=5, initial_sleep_time=10, 
                                                  error_type=SlackApiError)(_fetch_conversations_history)
        join_conversations_retry = retry(max_retries=5, initial_sleep_time=10, 
                                        error_type=SlackApiError)(_join_conversations)
        
        try:
            self._wait_api_call()
            result = fetch_conversations_history_retry(
                channel=channel_id,
                oldest=str(start_time.timestamp()),
                latest=str(end_time.timestamp()),
                limit=1000)
        except SlackApiError as error:
            if error.response['error'] == 'not_in_channel':
                self._wait_api_call()
                response = join_conversations_retry(channel=channel_id)
                if not response["ok"]:
                    print("Failed conversations_join()")
                    sys.exit(1)
                time.sleep(wait_time)
                self._wait_api_call()
                result = fetch_conversations_history_retry(
                    channel=channel_id,
                    oldest=str(start_time.timestamp()),
                    latest=str(end_time.timestamp()),
                    limit=1000)
            else:
                print(f"Error : {error}")
                return None
            
        while True:
            self._wait_api_call()
            result = fetch_conversations_history_retry(
                channel=channel_id,
                oldest=str(start_time.timestamp()),
                latest=str(end_time.timestamp()),
                limit=1000,
                cursor=next_cursor)
            
            messages_info.extend(result["messages"])
            next_cursor = result['response_metadata']['next_cursor']
            if not next_cursor:
                break
            
        # Filter for human messages only
        messages = list(filter(lambda m: "subtype" not in m, messages_info))

        if len(messages) < 1:
            return None

        messages_text = []
        for message in messages[::-1]:
            # Ignore bot messages and empty messages
            if "bot_id" in message or len(message["text"].strip()) == 0:
                continue

            # Get speaker name
            speaker_name = self.get_user_name(message["user"]) or "somebody"

            # Get message body from result dict.
            body_text = message["text"].replace("\n", "\\n")

            # Replace User IDs in a chat message text with user names.
            body_text = self.replace_user_id_with_name(body_text)

            # all channel id replace to "other channel"
            body_text = re.sub(r"<#[A-Z0-9]+>", " other channel ", body_text)

            messages_text.append(f"{speaker_name}: {body_text}")

        if len(messages_text) == 0:
            return None
        else:
            return messages_text

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

    def _get_users_info(self) -> list:
        """ Retrieve information about all users in the Slack workspace.

        Returns:
            list: A list of dictionaries containing information about each user,
                including their ID, name, and other metadata.

        Raises:
            SlackApiError: If an error occurs while attempting to retrieve the user information.

        Examples:
            >>> users = get_users_info()
            >>> print(users[0])
            {
                'id': 'U12345678',
                'name': 'alice',
                'real_name': 'Alice Smith',
                'email': 'alice@example.com',
                ...
            }
        """
        try:
            users = []
            next_cursor = None
            while True:
                self._wait_api_call()
                users_info = retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)(lambda: self.client.users_list(
                            cursor=next_cursor, limit=100))
                time.sleep(3)
                users.extend(users_info['members'])
                if users_info["response_metadata"]["next_cursor"]:
                    next_cursor = users_info["response_metadata"][
                        "next_cursor"]
                else:
                    break
            return users
        except SlackApiError as error:
            print(f"Error : {error}")
            sys.exit(1)

    def _get_channels_info(self) -> list:
        """ Retrieve information about all public channels in the Slack workspace.

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
        try:
            self._wait_api_call()
            result = retry(max_retries=5, initial_sleep_time=10, error_type=SlackApiError)(lambda: self.client.conversations_list(
                    types="public_channel", exclude_archived=True, limit=1000))
            channels_info = [
                channel for channel in result['channels']
                if not channel["is_archived"] and channel["is_channel"]
                    and (not channel["is_ext_shared"] and not channel["is_org_shared"] or ADD_SUMMARY_TAG in channel["purpose"]["value"])
                    and SKIP_SUMMARY_TAG not in channel["purpose"]["value"]
             ]
            channels_info = sort_by_numeric_prefix(channels_info,
                                                   get_key=lambda x: x["name"])
            return channels_info
        except SlackApiError as error:
            print(f"Error : {error}")
            sys.exit(1)

    def _wait_api_call(self):
        """ most of api call limit is 20 per minute """
        time.sleep(60 / 20)
