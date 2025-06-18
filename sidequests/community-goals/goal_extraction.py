"""
Goal Extraction and Deduplication System

This module provides a robust system for extracting student goals from JSON files,
with automatic deduplication and incremental processing capabilities.
It supports multiple JSON structures and maintains a master database.
"""

import json
import os
import glob
import hashlib
from typing import List, Dict, Any
from datetime import datetime


class GoalExtractor:
    """
    A class for extracting and managing student goals from JSON files.

    Features:
    - Automatic JSON file discovery
    - Multiple JSON structure support
    - Deduplication using message IDs and content hashes
    - Incremental processing
    - Master database management
    """

    def __init__(self, folder_path: str = None,
                 master_db_file: str = "master_goals_database.json"):
        """
        Initialize the GoalExtractor.
        Args:
            folder_path: Path to folder containing JSON files
            master_db_file: Name of the master database file
        """
        # If no folder_path specified, use the directory where this script is located
        if folder_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.folder_path = script_dir
        else:
            self.folder_path = folder_path

        # Create outputs directory for all generated files
        self.outputs_dir = os.path.join(self.folder_path, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Make database files use the outputs directory
        self.master_db_file = os.path.join(self.outputs_dir, master_db_file)
        self.processed_files_log = os.path.join(self.outputs_dir, "processed_files.json")

    def discover_json_files(self) -> List[str]:
        """
        Automatically discover all JSON files in the folder (except our output files).
        Returns:
            List of JSON file paths to process
        """
        # Get all JSON files in the main directory
        json_files = glob.glob(os.path.join(self.folder_path, "*.json"))

        # Exclude any files that are in the outputs directory or are legacy output files
        exclude_patterns = {
            "all_goals_export.json",
            "master_goals_database.json",
            "processed_files.json"
        }

        filtered_files = []
        for file_path in json_files:
            filename = os.path.basename(file_path)
            # Skip if it's an output file or in outputs directory
            if filename not in exclude_patterns and not file_path.startswith(self.outputs_dir):
                filtered_files.append(file_path)

        return filtered_files

    def load_existing_master_db(self) -> Dict[str, Any]:
        """
        Load existing master database or create new one.
        Returns:
            Dictionary containing the master database structure
        """
        if os.path.exists(self.master_db_file):
            try:
                with open(self.master_db_file, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_goals": 0,
                "unique_users": 0,
                "processed_files": []
            },
            "goals": {},  # Key: message_id, Value: goal data
            "user_index": {},  # Key: user_id, Value: list of message_ids
            "deduplication_index": {}  # Key: hash, Value: message_id
        }

    def load_processed_files_log(self) -> Dict[str, str]:
        """
        Load log of processed files with their modification times.
        Returns:
            Dictionary mapping filenames to modification timestamps
        """
        if os.path.exists(self.processed_files_log):
            try:
                with open(self.processed_files_log, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}

    def save_processed_files_log(self, log: Dict[str, str]) -> None:
        """
        Save log of processed files.
        Args:
            log: Dictionary mapping filenames to modification timestamps
        """
        with open(self.processed_files_log, 'w', encoding='utf-8') as file:
            json.dump(log, file, indent=2)

    def detect_json_structure(self, data: Dict[str, Any]) -> str:
        """
        Detect which JSON structure we're dealing with.
        Args:
            data: Parsed JSON data
        Returns:
            String indicating structure type: 'channels_response', 'single_channel', or 'unknown'
        """
        if "channels" in data and isinstance(data["channels"], list):
            return "channels_response"
        if "channel" in data and "messages" in data:
            return "single_channel"
        return "unknown"

    def create_deduplication_hash(self, user_id: str, text: str, created_at: str) -> str:
        """
        Create a hash for deduplication based on user + text + timestamp.
        Args:
            user_id: User identifier
            text: Message text content
            created_at: Timestamp when message was created
        Returns:
            MD5 hash string for deduplication
        """
        # Normalize text by removing extra whitespace and converting to lowercase
        normalized_text = " ".join(text.lower().strip().split())
        # Use date+time without microseconds for better deduplication
        hash_input = f"{user_id}|{normalized_text}|{created_at[:19]}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    def extract_goals_from_messages(self, messages: List[Dict],
                                    channel_name: str = "Unknown") -> List[Dict[str, Any]]:
        """
        Extract goals from a messages array.
        Args:
            messages: List of message dictionaries
            channel_name: Name of the channel these messages belong to
        Returns:
            List of extracted goal dictionaries
        """
        goals = []
        for message in messages:
            try:
                # Skip if message doesn't have required fields
                required_fields = ['id', 'text', 'user', 'created_at']
                if not all(key in message for key in required_fields):
                    continue

                goal_entry = {
                    'message_id': message['id'],
                    'user_id': message['user']['id'],
                    'user_name': message['user']['name'],
                    'goal_text': message['text'],
                    'goal_html': message.get('html', ''),
                    'created_at': message['created_at'],
                    'updated_at': message.get('updated_at', message['created_at']),
                    'channel_name': channel_name,
                    'submission_type': message.get('submission_type', 'reflection'),
                    'reaction_counts': message.get('reaction_counts', {}),
                    'dedup_hash': self.create_deduplication_hash(
                        message['user']['id'],
                        message['text'],
                        message['created_at']
                    )
                }
                goals.append(goal_entry)
            except (KeyError, TypeError) as error:
                print(f"Warning: Skipping malformed message - {error}")
                continue
        return goals

    def extract_from_channels_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract goals from channels-response.json format.
        Args:
            data: Parsed JSON data in channels response format
        Returns:
            List of extracted goals
        """
        all_goals = []
        for channel in data.get('channels', []):
            channel_name = channel.get('channel', {}).get('name', 'Unknown')
            messages = channel.get('messages', [])
            goals = self.extract_goals_from_messages(messages, channel_name)
            all_goals.extend(goals)
        return all_goals

    def extract_from_single_channel(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract goals from single channel format (firstq.json, etc.).

        Args:
            data: Parsed JSON data in single channel format

        Returns:
            List of extracted goals
        """
        channel_name = data.get('channel', {}).get('name', 'Unknown')
        messages = data.get('messages', [])
        return self.extract_goals_from_messages(messages, channel_name)

    def process_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Process a single JSON file and extract goals.
        Args:
            filepath: Path to the JSON file to process
        Returns:
            List of extracted goals from the file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            structure_type = self.detect_json_structure(data)
            if structure_type == "channels_response":
                return self.extract_from_channels_response(data)
            if structure_type == "single_channel":
                return self.extract_from_single_channel(data)
            print(f"Warning: Unknown JSON structure in {filepath}")
            return []

        except Exception as error:
            print(f"Error processing {filepath}: {error}")
            return []

    def add_goals_to_database(self, new_goals: List[Dict[str, Any]],
                              master_db: Dict[str, Any]) -> Dict[str, int]:
        """
        Add new goals to master database with deduplication.
        Args:
            new_goals: List of new goals to add
            master_db: Existing master database
        Returns:
            Dictionary with statistics about added/duplicate goals
        """
        stats = {"added": 0, "duplicates_id": 0, "duplicates_content": 0}

        for goal in new_goals:
            message_id = goal['message_id']
            dedup_hash = goal['dedup_hash']

            # Primary deduplication: Check if message ID already exists
            if message_id in master_db['goals']:
                stats["duplicates_id"] += 1
                continue

            # Secondary deduplication: Check if content hash already exists
            if dedup_hash in master_db['deduplication_index']:
                existing_id = master_db['deduplication_index'][dedup_hash]
                print(f"Content duplicate found: {message_id} matches existing {existing_id}")
                stats["duplicates_content"] += 1
                continue

            # Add to database
            master_db['goals'][message_id] = goal
            master_db['deduplication_index'][dedup_hash] = message_id

            # Update user index
            user_id = goal['user_id']
            if user_id not in master_db['user_index']:
                master_db['user_index'][user_id] = []
            master_db['user_index'][user_id].append(message_id)

            stats["added"] += 1

        return stats

    def update_master_database(self) -> Dict[str, Any]:
        """
        Main function to update the master database with all JSON files.
        Returns:
            Updated master database
        """
        print("🔍 Discovering JSON files...")
        json_files = self.discover_json_files()
        file_basenames = [os.path.basename(f) for f in json_files]
        print(f"Found {len(json_files)} JSON files: {file_basenames}")

        print("\n📖 Loading existing database...")
        master_db = self.load_existing_master_db()
        processed_files_log = self.load_processed_files_log()

        total_stats = {"added": 0, "duplicates_id": 0, "duplicates_content": 0}
        new_files_processed = []

        for json_file in json_files:
            filename = os.path.basename(json_file)
            file_mtime = str(os.path.getmtime(json_file))

            # Check if file was already processed and hasn't changed
            if (filename in processed_files_log and
                processed_files_log[filename] == file_mtime):
                print(f"⏭️  Skipping {filename} (already processed, no changes)")
                continue

            print(f"\n🔄 Processing {filename}...")
            new_goals = self.process_json_file(json_file)

            if new_goals:
                file_stats = self.add_goals_to_database(new_goals, master_db)
                print(f"   ✅ Added: {file_stats['added']}, "
                      f"ID Duplicates: {file_stats['duplicates_id']}, "
                      f"Content Duplicates: {file_stats['duplicates_content']}")

                # Update totals
                for key in total_stats:
                    total_stats[key] += file_stats[key]

                # Mark file as processed
                processed_files_log[filename] = file_mtime
                new_files_processed.append(filename)
            else:
                print(f"   ⚠️  No goals extracted from {filename}")

        # Update metadata
        master_db['metadata']['last_updated'] = datetime.now().isoformat()
        master_db['metadata']['total_goals'] = len(master_db['goals'])
        master_db['metadata']['unique_users'] = len(master_db['user_index'])
        master_db['metadata']['processed_files'] = list(processed_files_log.keys())

        # Save everything
        print("\n💾 Saving master database...")
        with open(self.master_db_file, 'w', encoding='utf-8') as file:
            json.dump(master_db, file, indent=2, ensure_ascii=False)

        self.save_processed_files_log(processed_files_log)

        self._print_final_summary(master_db, total_stats, new_files_processed)

        return master_db

    def _print_final_summary(self, master_db: Dict[str, Any],
                             total_stats: Dict[str, int],
                             new_files_processed: List[str]) -> None:
        """
        Print final summary of the processing run.
        Args:
            master_db: The master database
            total_stats: Statistics from this processing run
            new_files_processed: List of files processed in this run
        """
        print("\n📊 FINAL SUMMARY:")
        print(f"   Total goals in database: {master_db['metadata']['total_goals']}")
        print(f"   Unique users: {master_db['metadata']['unique_users']}")
        print(f"   New goals added: {total_stats['added']}")
        print(f"   ID duplicates skipped: {total_stats['duplicates_id']}")
        print(f"   Content duplicates skipped: {total_stats['duplicates_content']}")
        print(f"   Files processed this run: {len(new_files_processed)}")

    def export_goals_list(self, output_file: str = "all_goals_export.json") -> List[Dict[str, Any]]:
        """
        Export just the goals in a clean list format.
        Args:
            output_file: Name of the output file
        Returns:
            List of all goals
        """
        master_db = self.load_existing_master_db()
        goals_list = list(master_db.get('goals', {}).values())

        # Ensure output file is in the outputs directory
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.outputs_dir, output_file)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(goals_list, file, indent=2, ensure_ascii=False)

        print(f"📄 Exported {len(goals_list)} goals to {output_file}")
        return goals_list

    def get_statistics(self) -> None:
        """Get and print statistics about the goals database."""
        master_db = self.load_existing_master_db()
        goals = master_db.get('goals', {})

        if not goals:
            print("No goals in database yet. Run update_master_database() first.")
            return

        # Basic stats
        total_goals = len(goals)
        unique_users = len(master_db.get('user_index', {}))

        # Channel distribution and keyword analysis
        channel_counts = {}
        keyword_counts = {'RL': 0, 'reinforcement': 0, 'agent': 0, 'MCP': 0}

        for goal in goals.values():
            channel = goal.get('channel_name', 'Unknown')
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

            # Keyword analysis
            text_lower = goal.get('goal_text', '').lower()
            for keyword in keyword_counts:
                if keyword.lower() in text_lower:
                    keyword_counts[keyword] += 1

        self._print_statistics(total_goals, unique_users, channel_counts, keyword_counts)

    def _print_statistics(self, total_goals: int, unique_users: int,
                          channel_counts: Dict[str, int],
                          keyword_counts: Dict[str, int]) -> None:
        """
        Print formatted statistics.
        Args:
            total_goals: Total number of goals
            unique_users: Number of unique users
            channel_counts: Distribution of goals by channel
            keyword_counts: Count of goals containing keywords
        """
        print("\n📈 DATABASE STATISTICS:")
        print(f"   Total goals: {total_goals}")
        print(f"   Unique users: {unique_users}")
        print(f"   Average goals per user: {total_goals/unique_users:.1f}")

        print("\n📊 CHANNEL DISTRIBUTION:")
        for channel, count in sorted(channel_counts.items()):
            print(f"   {channel}: {count} goals")

        print("\n🔤 KEYWORD ANALYSIS:")
        for keyword, count in keyword_counts.items():
            percentage = (count/total_goals)*100
            print(f"   '{keyword}': {count} goals ({percentage:.1f}%)")


def main() -> None:
    """Main function to run the goal extraction process."""
    extractor = GoalExtractor()

    print("🚀 Starting Goal Extraction Process...")
    extractor.update_master_database()

    print("\n📋 Exporting clean goals list...")
    extractor.export_goals_list()

    print("\n📊 Database Statistics:")
    extractor.get_statistics()


if __name__ == "__main__":
    main()
