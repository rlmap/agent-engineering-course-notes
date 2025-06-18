"""
Goal Categorization

This script categorizes goals based on keywords and patterns.
It uses a set of predefined categories and keywords to categorize the goals.
It then creates a visualizations of the categories and the goals.
"""

import json
from typing import List, Dict, Set
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

def load_goals(file_path: str) -> List[Dict]:
    """Load goals from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def categorize_goal(goal_text: str) -> Set[str]:
    """
    Categorize a goal text based on keywords and patterns.
    Returns a set of categories that apply to this goal.
    """
    categories = set()
    goal_lower = goal_text.lower()

    # Define keyword patterns for each category
    category_patterns = {
        'Foundation Building': [
            'learn', 'understand', 'foundation', 'knowledge', 'concepts', 'theory', 
            'theoretical', 'grasp', 'soak up', 'penetrate', 'deepen', 'gain familiarity',
            'bootstrap', 'refresher', 'refresh'
        ],

        'Hands-on Technical Skills': [
            'hands-on', 'practical', 'experience', 'implement', 'fine-tun', 'grpo', 
            'training', 'deploy', 'code', 'technical', 'setup', 'build', 'development',
            'in-the-trenches', 'apply rl', 'rl fine-tuning'
        ],

        'Prototyping': [
            'build', 'prototype', 'poc', 'proof-of-concept', 'ship', 'create', 
            'develop', 'project', 'working prototype', 'deliver', 'stand up',
            'implement', 'system'
        ],

        'Enterprise (Production) Deployment': [
            'production', 'prod', 'enterprise', 'organization', 'org', 'company',
            'production-ready', 'staging', 'deploy', 'internal', 'business'
        ],

        'Domain-Specific Applications': [
            'banking', 'medical', 'smart-lights', 'home assistant', 'regulation',
            'lean 4', 'proof verification', 'smart home', 'fdic', 'diagnosis',
            'upskilling', 'coach', 'puzzles', 'specific', 'domain'
        ],

        'Best Practices': [
            'best practices', 'when to use', 'tradeoffs', 'trade-offs', 'patterns',
            'architecture', 'design', 'decision', 'choose', 'bottlenecks',
            'challenges', 'landscape', 'frontier', 'scaffolding'
        ],

        'Evaluation + Optimization': [
            'evaluat', 'reward', 'feedback', 'metrics', 'performance', 'optimize',
            'judge', 'score', 'eval', 'test', 'measure', 'improve'
        ],

        'Networking': [
            'network', 'collaboration', 'collab', 'connect', 'like-minds',
            'exchange', 'mutual projects', 'folks', 'cohort', 'community',
            'learn from', 'battle-tested', 'spark'
        ],

        'Career': [
            'career', 'professional', 'interview', 'job', 'role', 'work',
            'engineering org', 'day-to-day', 'confidence', 'proficient'
        ],

        'Innovation': [
            'research', 'frontier', 'cutting edge', 'improve on papers', 'novel',
            'advance', 'next echelon', 'sota', 'state-of-the-art', 'from scratch'
        ]
    }

    # Check each category
    for category, keywords in category_patterns.items():
        for keyword in keywords:
            if keyword in goal_lower:
                categories.add(category)
                break  # Found a match for this category, move to next

    # If no categories found, classify as general learning
    if not categories:
        categories.add('Foundation Building')

    return categories

def analyze_goals(goals: List[Dict]) -> Dict[str, int]:
    """Analyze goals and return category counts."""
    category_counts = defaultdict(int)
    goal_categories = {}

    for goal in goals:
        goal_text = goal['goal_text']
        user_name = goal['user_name']
        categories = categorize_goal(goal_text)
        goal_categories[user_name] = categories

        # Count each category
        for category in categories:
            category_counts[category] += 1

    return dict(category_counts), goal_categories

def create_visualizations(category_counts: Dict[str, int], goal_categories: Dict[str, Set[str]]):
    """Create various visualizations of the goal categories."""

    # Set up the style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Agent Engineering Goals', fontsize=18, fontweight='bold')

    # 1. Horizontal bar chart
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # Sort by count for better visualization
    sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
    sorted_categories, sorted_counts = zip(*sorted_data)

    bars = ax1.barh(range(len(sorted_categories)), sorted_counts, color=sns.color_palette("husl", len(sorted_categories)))
    ax1.set_yticks(range(len(sorted_categories)))
    ax1.set_yticklabels([cat.replace(' & ', '\n& ') for cat in sorted_categories], fontsize=11)
    ax1.set_xlabel('Number of Goals', fontsize=12)
    ax1.set_title('Goals by Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=12, fontweight='bold')

    # 2. Pie chart with improved label positioning
    def autopct_func(pct):
        # Only show percentage if it's above 5% to avoid clutter
        return f'{pct:.1f}%' if pct > 5 else ''

    wedges, texts, autotexts = ax2.pie(sorted_counts,
                                       labels=[cat.replace(' & ', '\n& ') for cat in sorted_categories],
                                       autopct=autopct_func,
                                       startangle=90,
                                       textprops={'fontsize': 10},
                                       pctdistance=0.85,  # Move percentages closer to center
                                       labeldistance=1.1)  # Move labels further from center

    # Improve text positioning for better readability
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('black')

    ax2.set_title('Goals Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save the plot
    plt.savefig('sidequests/community-goals/outputs/goal_categories_analysis.png', dpi=300, bbox_inches='tight')

    # Create a detailed breakdown table
    print("\n" + "="*60)
    print("DETAILED CATEGORY BREAKDOWN")
    print("="*60)

    total_goals = len(goal_categories)
    print(f"Total participants: {total_goals}")
    print(f"Total category assignments: {sum(category_counts.values())}")
    print(f"Average categories per goal: {sum(category_counts.values()) / total_goals:.2f}")

    print(f"\n{'Category':<35} {'Count':<8} {'Percentage':<12}")
    print("-" * 55)

    for category, count in sorted_data:
        percentage = (count / total_goals) * 100
        print(f"{category:<35} {count:<8} {percentage:>8.1f}%")

def save_detailed_analysis(goal_categories: Dict[str, Set[str]], goals: List[Dict]):
    """Save detailed analysis to JSON file."""

    # Create detailed breakdown
    detailed_analysis = {
        'summary': {
            'total_participants': len(goal_categories),
            'total_category_assignments': sum(len(cats) for cats in goal_categories.values()),
            'average_categories_per_goal': sum(len(cats) for cats in goal_categories.values()) / len(goal_categories)
        },
        'participants': []
    }

    # Add participant details
    for goal in goals:
        user_name = goal['user_name']
        categories = list(goal_categories[user_name])

        detailed_analysis['participants'].append({
            'user_name': user_name,
            'user_id': goal['user_id'],
            'goal_text': goal['goal_text'],
            'categories': categories,
            'category_count': len(categories),
            'created_at': goal['created_at']
        })

    # Save to file
    with open('sidequests/community-goals/outputs/detailed_goal_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)

    print("\nDetailed analysis saved to: sidequests/community-goals/outputs/detailed_goal_analysis.json")

def main():
    """Main function to run the analysis."""

    # Load goals
    goals = load_goals('sidequests/community-goals/outputs/all_goals_export.json')
    print(f"Loaded {len(goals)} goals from all_goals_export.json")

    # Analyze goals
    category_counts, goal_categories = analyze_goals(goals)

    # Create visualizations
    create_visualizations(category_counts, goal_categories)

    # Save detailed analysis
    save_detailed_analysis(goal_categories, goals)

    print("\nAnalysis complete! Check the generated visualizations and detailed analysis file.")

if __name__ == "__main__":
    main()
