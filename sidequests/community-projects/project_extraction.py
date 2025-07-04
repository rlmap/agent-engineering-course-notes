"""
Project Extraction and Deduplication System

This module provides a robust system for extracting student projects from JSON files,
with automatic deduplication and incremental processing capabilities.
It supports multiple JSON structures and maintains a master database.
"""

import re
import os
import json
import glob
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any
from datetime import datetime


class ProjectExtractor:
    """
    A class for extracting and managing student projects from JSON files.

    Features:
    - Automatic JSON file discovery
    - Multiple JSON structure support
    - Deduplication using message IDs and content hashes
    - Incremental processing
    - Master database management
    """

    def __init__(self, folder_path: str = None,
                 master_db_file: str = "master_projects_database.json"):
        """
        Initialize the ProjectExtractor.
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
            "all_projects_export.json",
            "master_projects_database.json",
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
                "total_projects": 0,
                "unique_users": 0,
                "processed_files": []
            },
            "projects": {},  # Key: message_id, Value: project data
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

    def extract_github_urls(self, text: str, attachments: List[Dict] = None) -> List[str]:
        """
        Extract and normalize GitHub URLs from project text and attachments.
        Args:
            text: Message text content
            attachments: List of message attachments
        Returns:
            List of cleaned and deduplicated GitHub URLs
        """
        github_urls = []

        # Extract from text using multiple strategies
        if "github.com" in text.lower():

            # Strategy 1: Extract markdown links [text](url) - both github.com and gist.github.com
            markdown_pattern = r'\[([^\]]+)\]\((https://(?:gist\.)?github\.com[^\)]+)\)'
            markdown_matches = re.findall(markdown_pattern, text, re.IGNORECASE)
            for _, url in markdown_matches:
                github_urls.append(url)

            # Strategy 2: Extract plain URLs - both github.com and gist.github.com
            url_pattern = r'https://(?:gist\.)?github\.com[^\s\]\)]*'
            plain_matches = re.findall(url_pattern, text, re.IGNORECASE)
            github_urls.extend(plain_matches)

            # Strategy 3: Fallback word-by-word (cleaned up)
            words = text.split()
            for word in words:
                if "github.com" in word.lower():
                    # Clean up common markdown artifacts and punctuation
                    cleaned = re.sub(r'[\[\]()]+.*$', '', word)  # Remove ]( artifacts
                    cleaned = cleaned.strip('.,!?;:')
                    if cleaned.startswith('http') and 'github.com' in cleaned:
                        github_urls.append(cleaned)

        # Extract from attachments
        if attachments:
            for attachment in attachments:
                for field in ['title_link', 'og_scrape_url']:
                    url = attachment.get(field, '')
                    if url and ('github.com' in url.lower() or 'gist.github.com' in url.lower()):
                        github_urls.append(url)

        # Normalize and deduplicate URLs
        return self._normalize_github_urls(github_urls)

    def _normalize_github_urls(self, urls: List[str]) -> List[str]:
        """
        Normalize GitHub URLs and remove duplicates.
        Args:
            urls: List of raw GitHub URLs
        Returns:
            List of normalized, deduplicated URLs
        """
        normalized_urls = set()

        for url in urls:
            if not url or not isinstance(url, str):
                continue

            # Skip malformed URLs with markdown artifacts
            if '](' in url or url.count('http') > 1:
                continue

            # Clean up the URL
            url = url.strip()

            # Remove trailing punctuation
            url = re.sub(r'[.,!?;:]+$', '', url)

            # Ensure it starts with http
            if not url.startswith('http'):
                continue

            try:
                parsed = urlparse(url)

                # Handle both github.com and gist.github.com
                if parsed.netloc.lower() not in ['github.com', 'gist.github.com']:
                    continue

                # Normalize the path
                path = parsed.path.rstrip('/')

                # Reconstruct clean URL
                if parsed.netloc.lower() == 'gist.github.com':
                    clean_url = f"https://gist.github.com{path}"
                else:
                    clean_url = f"https://github.com{path}"

                # Only keep URLs that look like valid GitHub repos/gists
                if self._is_valid_github_url(clean_url):
                    normalized_urls.add(clean_url)

            except Exception:
                # Skip invalid URLs
                continue

        return sorted(list(normalized_urls))

    def _is_valid_github_url(self, url: str) -> bool:
        """
        Check if a GitHub URL looks valid (repo, gist, etc.).
        Args:
            url: Normalized GitHub URL
        Returns:
            True if URL appears to be a valid GitHub resource
        """

        # Handle gist URLs
        if url.startswith('https://gist.github.com/'):
            path = url.replace('https://gist.github.com/', '')
            # Gist URLs should have at least username/gistid format
            return len(path) > 10 and '/' in path

        # Handle regular github URLs
        if not url.startswith('https://github.com/'):
            return False

        path = url.replace('https://github.com/', '')

        # Skip if too short or just github.com
        if len(path) < 3:
            return False

        # Valid patterns for github.com:
        # - username/repo
        # - username/repo/tree/branch
        # - username/repo/blob/branch/file

        # Check for valid GitHub repo patterns
        username_repo_pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+'

        if re.match(username_repo_pattern, path):
            return True

        return False

    def extract_projects_from_messages(self, messages: List[Dict],
                                      channel_name: str = "Unknown") -> List[Dict[str, Any]]:
        """
        Extract projects from a messages array.
        Args:
            messages: List of message dictionaries
            channel_name: Name of the channel these messages belong to
        Returns:
            List of extracted project dictionaries
        """
        projects = []
        for message in messages:
            try:
                # Skip if message doesn't have required fields
                required_fields = ['id', 'text', 'user', 'created_at']
                if not all(key in message for key in required_fields):
                    continue

                # Filter 1: Only process messages from "Prototype an Agent" channel
                if channel_name != "Prototype an Agent":
                    continue

                # Filter 2: Only process messages with submission_type "project"
                submission_type = message.get('submission_type', 'unknown')
                if submission_type != 'project':
                    continue

                # Extract GitHub URLs
                github_urls = self.extract_github_urls(
                    message['text'], 
                    message.get('attachments', [])
                )

                project_entry = {
                    'message_id': message['id'],
                    'user_id': message['user']['id'],
                    'user_name': message['user']['name'],
                    'project_text': message['text'],
                    'project_html': message.get('html', ''),
                    'github_urls': github_urls,
                    'created_at': message['created_at'],
                    'updated_at': message.get('updated_at', message['created_at']),
                    'channel_name': channel_name,
                    'project_type': message.get('type', 'regular'),
                    'submission_type': submission_type,
                    'attachments': message.get('attachments', []),
                    'reaction_counts': message.get('reaction_counts', {}),
                    'reaction_scores': message.get('reaction_scores', {}),
                    'reply_count': message.get('reply_count', 0),
                    'dedup_hash': self.create_deduplication_hash(
                        message['user']['id'],
                        message['text'],
                        message['created_at']
                    )
                }
                projects.append(project_entry)
            except (KeyError, TypeError) as error:
                print(f"Warning: Skipping malformed message - {error}")
                continue
        return projects

    def extract_from_channels_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract projects from channels-response.json format.
        Args:
            data: Parsed JSON data in channels response format
        Returns:
            List of extracted projects
        """
        all_projects = []
        for channel in data.get('channels', []):
            channel_name = channel.get('channel', {}).get('name', 'Unknown')
            messages = channel.get('messages', [])
            projects = self.extract_projects_from_messages(messages, channel_name)
            all_projects.extend(projects)
        return all_projects

    def extract_from_single_channel(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract projects from single channel format (prototype_response0.json, etc.).

        Args:
            data: Parsed JSON data in single channel format

        Returns:
            List of extracted projects
        """
        channel_name = data.get('channel', {}).get('name', 'Unknown')
        messages = data.get('messages', [])
        return self.extract_projects_from_messages(messages, channel_name)

    def process_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Process a single JSON file and extract projects.
        Args:
            filepath: Path to the JSON file to process
        Returns:
            List of extracted projects from the file
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

    def add_projects_to_database(self, new_projects: List[Dict[str, Any]],
                                master_db: Dict[str, Any]) -> Dict[str, int]:
        """
        Add new projects to master database with deduplication.
        Args:
            new_projects: List of new projects to add
            master_db: Existing master database
        Returns:
            Dictionary with statistics about added/duplicate projects
        """
        stats = {"added": 0, "duplicates_id": 0, "duplicates_content": 0}

        for project in new_projects:
            message_id = project['message_id']
            dedup_hash = project['dedup_hash']

            # Primary deduplication: Check if message ID already exists
            if message_id in master_db['projects']:
                stats["duplicates_id"] += 1
                continue

            # Secondary deduplication: Check if content hash already exists
            if dedup_hash in master_db['deduplication_index']:
                existing_id = master_db['deduplication_index'][dedup_hash]
                print(f"Content duplicate found: {message_id} matches existing {existing_id}")
                stats["duplicates_content"] += 1
                continue

            # Add to database
            master_db['projects'][message_id] = project
            master_db['deduplication_index'][dedup_hash] = message_id

            # Update user index
            user_id = project['user_id']
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
            new_projects = self.process_json_file(json_file)

            if new_projects:
                file_stats = self.add_projects_to_database(new_projects, master_db)
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
                print(f"   ⚠️  No projects extracted from {filename}")

        # Update metadata
        master_db['metadata']['last_updated'] = datetime.now().isoformat()
        master_db['metadata']['total_projects'] = len(master_db['projects'])
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
        print(f"   Total projects in database: {master_db['metadata']['total_projects']}")
        print(f"   Unique users: {master_db['metadata']['unique_users']}")
        print(f"   New projects added: {total_stats['added']}")
        print(f"   ID duplicates skipped: {total_stats['duplicates_id']}")
        print(f"   Content duplicates skipped: {total_stats['duplicates_content']}")
        print(f"   Files processed this run: {len(new_files_processed)}")

    def export_projects_list(self, output_file: str = "all_projects_export.json") -> List[Dict[str, Any]]:
        """
        Export just the projects in a clean list format.
        Args:
            output_file: Name of the output file
        Returns:
            List of all projects
        """
        master_db = self.load_existing_master_db()
        projects_list = list(master_db.get('projects', {}).values())

        # Ensure output file is in the outputs directory
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.outputs_dir, output_file)

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(projects_list, file, indent=2, ensure_ascii=False)

        print(f"📄 Exported {len(projects_list)} projects to {output_file}")
        return projects_list

    def get_statistics(self) -> None:
        """Get and print statistics about the projects database."""
        master_db = self.load_existing_master_db()
        projects = master_db.get('projects', {})

        if not projects:
            print("No projects in database yet. Run update_master_database() first.")
            return

        # Basic stats
        total_projects = len(projects)
        unique_users = len(master_db.get('user_index', {}))

        # Channel distribution and analysis
        channel_counts = {}
        submission_type_counts = {}
        github_count = 0
        keyword_counts = {'agent': 0, 'AI': 0, 'machine learning': 0, 'deep learning': 0, 'LLM': 0}

        for project in projects.values():
            channel = project.get('channel_name', 'Unknown')
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
            # Track submission types
            submission_type = project.get('submission_type', 'unknown')
            submission_type_counts[submission_type] = submission_type_counts.get(submission_type, 0) + 1

            # GitHub URL analysis
            if project.get('github_urls'):
                github_count += 1

            # Keyword analysis
            text_lower = project.get('project_text', '').lower()
            for keyword in keyword_counts:
                if keyword.lower() in text_lower:
                    keyword_counts[keyword] += 1

        self._print_statistics(total_projects, unique_users, channel_counts, 
                              keyword_counts, github_count, submission_type_counts)

    def _print_statistics(self, total_projects: int, unique_users: int,
                          channel_counts: Dict[str, int],
                          keyword_counts: Dict[str, int],
                          github_count: int,
                          submission_type_counts: Dict[str, int] = None) -> None:
        """
        Print formatted statistics.
        Args:
            total_projects: Total number of projects
            unique_users: Number of unique users
            channel_counts: Distribution of projects by channel
            keyword_counts: Count of projects containing keywords
            github_count: Number of projects with GitHub URLs
            submission_type_counts: Distribution of projects by submission type
        """
        print("\n📈 DATABASE STATISTICS:")
        print(f"   Total projects: {total_projects}")
        print(f"   Unique users: {unique_users}")
        print(f"   Average projects per user: {total_projects/unique_users:.1f}")
        print(f"   Projects with GitHub URLs: {github_count} ({(github_count/total_projects)*100:.1f}%)")

        print("\n📊 CHANNEL DISTRIBUTION:")
        for channel, count in sorted(channel_counts.items()):
            print(f"   {channel}: {count} projects")

        if submission_type_counts:
            print("\n📝 SUBMISSION TYPE DISTRIBUTION:")
            for submission_type, count in sorted(submission_type_counts.items()):
                print(f"   {submission_type}: {count} projects")

        print("\n🔤 KEYWORD ANALYSIS:")
        for keyword, count in keyword_counts.items():
            percentage = (count/total_projects)*100
            print(f"   '{keyword}': {count} projects ({percentage:.1f}%)")


def main() -> None:
    """Main function to run the project extraction process."""
    extractor = ProjectExtractor()

    print("🚀 Starting Project Extraction Process...")
    extractor.update_master_database()

    print("\n📋 Exporting clean projects list...")
    extractor.export_projects_list()

    print("\n📊 Database Statistics:")
    extractor.get_statistics()


if __name__ == "__main__":
    main()
