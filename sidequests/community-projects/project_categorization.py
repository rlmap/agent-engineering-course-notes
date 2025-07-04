"""
Project Categorization and Analysis

This script categorizes student projects based on implementation patterns, technology stacks,
and project characteristics. It provides detailed analysis including quality metrics,
technology usage, and timeline trends.
"""

import re
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_projects(file_path: str) -> List[Dict]:
    """Load projects from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_project(project_text: str, github_urls: List[str] = None) -> Set[str]:
    """
    Categorize a project based on implementation patterns and described features.
    Returns a set of granular categories that apply to this project.
    """
    categories = set()
    text_lower = project_text.lower()

    # Granular category patterns
    category_patterns = {
        # Multi-Agent Systems
        'Multi-Agent Orchestration': [
            'multi-agent', 'multiple agents', 'agent orchestration', 'agent coordination',
            'orchestration pattern', 'agent delegation', 'collaborative agents', 'agent workflow'
        ],

        'Multi-Agent Delegation': [
            'agent delegation', 'delegate to', 'delegating agent', 'agent handoff',
            'specialized agents', 'agent specialization', 'agent collaboration'
        ],

        # Single Agent Applications  
        'Single Agent Application': [
            'single agent', 'one agent', 'main agent', 'primary agent',
            'individual agent', 'standalone agent'
        ],

        # RL Specific
        'Reinforcement Learning': [
            'reinforcement learning', 'rl fine-tuning', 'rl training', 'policy',
            'reward function', 'rl agent', 'policy gradient', 'q-learning'
        ],

        # Data & Analysis
        'Data Processing/ETL': [
            'data processing', 'etl', 'data pipeline', 'data extraction',
            'data transformation', 'data ingestion', 'batch processing'
        ],

        'Classification/Analysis': [
            'classification', 'classifier', 'analysis', 'categorization',
            'pattern recognition', 'sentiment analysis', 'text analysis'
        ],

        # Web Applications
        'Web Application': [
            'web app', 'web application', 'frontend', 'ui', 'user interface',
            'dashboard', 'streamlit', 'gradio', 'web interface'
        ],

        'API/Backend Service': [
            'api', 'backend', 'service', 'microservice', 'rest api',
            'fastapi', 'flask', 'server', 'endpoint'
        ],

        # Domain Specific
        'Finance/Trading': [
            'trading', 'financial', 'finance', 'banking', 'investment',
            'portfolio', 'market', 'stock', 'cryptocurrency', 'defi', 'rfq'
        ],

        'Healthcare/Medical': [
            'healthcare', 'medical', 'health', 'diagnosis', 'patient',
            'clinical', 'disease', 'treatment'
        ],

        'Education/Learning': [
            'education', 'educational', 'learning', 'teaching', 'tutor',
            'student', 'course', 'training'
        ],

        'Home/IoT': [
            'home assistant', 'smart home', 'iot', 'smart lights', 'automation',
            'home automation', 'smart device'
        ],

        # Technical Categories
        'Research/Experimentation': [
            'research', 'experiment', 'experimental', 'novel approach',
            'investigation', 'prototype', 'proof of concept', 'poc'
        ],

        'Evaluation/Benchmarking': [
            'evaluation', 'benchmark', 'testing', 'performance test',
            'evaluation framework', 'metrics', 'assessment'
        ],

        'Infrastructure/Tooling': [
            'infrastructure', 'tooling', 'framework', 'library',
            'developer tool', 'utility', 'helper', 'template'
        ],

        'Document Processing': [
            'document', 'pdf', 'text processing', 'document analysis',
            'file processing', 'document extraction'
        ],

        'Conversational AI': [
            'chatbot', 'conversational', 'dialogue', 'chat', 'assistant',
            'conversation', 'interactive'
        ]
    }

    # Check each category
    for category, keywords in category_patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                categories.add(category)
                break

    # Special logic for single vs multi-agent
    if any(cat.startswith('Multi-Agent') for cat in categories):
        categories.discard('Single Agent Application')
    elif not any(cat.startswith('Multi-Agent') for cat in categories) and 'agent' in text_lower:
        categories.add('Single Agent Application')

    return categories


def detect_technology_stack(project_text: str, attachments: List[Dict] = None) -> Set[str]:
    """
    Detect specific technologies and frameworks mentioned in the project.
    """
    technologies = set()
    text_lower = project_text.lower()

    # Framework/Library patterns
    tech_patterns = {
        # AI/ML Frameworks
        'PydanticAI': ['pydanticai', 'pydantic ai'],
        'LangChain': ['langchain', 'lang chain'],
        'LlamaIndex': ['llamaindex', 'llama index'],
        'Transformers': ['transformers', 'huggingface', 'hugging face'],

        # API Providers
        'OpenAI': ['openai', 'gpt-4', 'gpt-3', 'chatgpt'],
        'Anthropic': ['anthropic', 'claude'],
        'Google AI': ['google ai', 'gemini', 'palm'],

        # Web Frameworks
        'FastAPI': ['fastapi', 'fast api'],
        'Flask': ['flask'],
        'Streamlit': ['streamlit'],
        'Gradio': ['gradio'],
        'Django': ['django'],

        # Data & ML
        'Pandas': ['pandas'],
        'NumPy': ['numpy'],
        'Scikit-learn': ['scikit-learn', 'sklearn'],
        'PyTorch': ['pytorch', 'torch'],
        'TensorFlow': ['tensorflow'],

        # Databases
        'PostgreSQL': ['postgresql', 'postgres'],
        'MongoDB': ['mongodb', 'mongo'],
        'SQLite': ['sqlite'],
        'Redis': ['redis'],

        # Observability/Monitoring
        'Logfire': ['logfire'],
        'Weights & Biases': ['wandb', 'weights and biases', 'weights & biases'],
        'MLflow': ['mlflow'],

        # Cloud/Infrastructure
        'Docker': ['docker'],
        'AWS': ['aws', 'amazon web services'],
        'Google Cloud': ['google cloud', 'gcp'],
        'Azure': ['azure'],
        'Vercel': ['vercel'],
        'Heroku': ['heroku'],

        # Other Tools
        'React': ['react', 'react.js'],
        'Next.js': ['next.js', 'nextjs'],
        'TypeScript': ['typescript'],
        'Python': ['python'],
        'JavaScript': ['javascript', 'js'],
        'Node.js': ['node.js', 'nodejs']
    }

    for tech, patterns in tech_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                technologies.add(tech)
                break

    return technologies


def calculate_project_quality_score(project: Dict) -> Dict[str, any]:
    """
    Calculate various quality indicators for a project.
    Not based on popularity but on objective characteristics.
    """
    text = project.get('project_text', '')
    github_urls = project.get('github_urls', [])
    attachments = project.get('attachments', [])

    quality_metrics = {
        'has_github': len(github_urls) > 0,
        'github_count': len(github_urls),
        'description_length': len(text),
        'has_attachments': len(attachments) > 0,
        'detailed_description': len(text) > 500,  # Substantial description
        'mentions_technical_details': bool(re.search(r'\b(architecture|implementation|algorithm|framework|api|database)\b', text.lower())),
        'mentions_challenges': bool(re.search(r'\b(challenge|problem|roadblock|difficulty|issue)\b', text.lower())),
        'mentions_results': bool(re.search(r'\b(result|performance|accuracy|improvement|success)\b', text.lower())),
        'mentions_future_work': bool(re.search(r'\b(future|next|plan|improve|enhance|todo)\b', text.lower())),
        'code_references': bool(re.search(r'\b(code|implementation|repository|repo|github)\b', text.lower())),
        'structured_presentation': bool(re.search(r'\*\*.*\*\*', text)),  # Uses markdown formatting
    }

    # Calculate composite scores
    documentation_score = sum([
        quality_metrics['detailed_description'],
        quality_metrics['mentions_technical_details'],
        quality_metrics['structured_presentation'],
        quality_metrics['mentions_challenges'],
        quality_metrics['mentions_results']
    ]) / 5

    completeness_score = sum([
        quality_metrics['has_github'],
        quality_metrics['detailed_description'],
        quality_metrics['code_references'],
        quality_metrics['mentions_technical_details']
    ]) / 4

    quality_metrics['documentation_score'] = documentation_score
    quality_metrics['completeness_score'] = completeness_score
    quality_metrics['overall_quality'] = (documentation_score + completeness_score) / 2

    return quality_metrics


def analyze_projects(projects: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Comprehensive analysis of projects.
    Returns category counts, technology usage, quality metrics, and detailed project data.
    """
    category_counts = defaultdict(int)
    technology_counts = defaultdict(int)
    project_analysis = {}
    timeline_data = []

    for project in projects:
        project_text = project['project_text']
        user_name = project['user_name']
        github_urls = project.get('github_urls', [])
        created_at = project['created_at']

        # Categorize project
        categories = categorize_project(project_text, github_urls)

        # Detect technologies
        technologies = detect_technology_stack(project_text, project.get('attachments', []))

        # Calculate quality metrics
        quality_metrics = calculate_project_quality_score(project)

        # Store analysis
        project_analysis[user_name] = {
            'categories': list(categories),
            'technologies': list(technologies),
            'quality_metrics': quality_metrics,
            'created_at': created_at,
            'github_urls': github_urls
        }

        # Count categories and technologies
        for category in categories:
            category_counts[category] += 1
        for tech in technologies:
            technology_counts[tech] += 1

        # Timeline data
        timeline_data.append({
            'date': created_at,
            'categories': list(categories),
            'technologies': list(technologies),
            'user': user_name
        })

    return dict(category_counts), dict(technology_counts), project_analysis, timeline_data


def create_comprehensive_visualizations(category_counts: Dict, technology_counts: Dict,
                                        project_analysis: Dict, timeline_data: List[Dict], projects: List[Dict]):
    """Create comprehensive visualizations of project analysis."""

    # Set up style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Student Project Analysis - Comprehensive Overview', fontsize=20, fontweight='bold')

    # 1. Project Categories (Top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if category_counts:
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
        sorted_cats, sorted_counts = zip(*sorted_data[:10])  # Top 10

        bars = ax1.barh(range(len(sorted_cats)), sorted_counts)
        ax1.set_yticks(range(len(sorted_cats)))
        ax1.set_yticklabels([cat.replace(' ', '\n') for cat in sorted_cats], fontsize=9)
        ax1.set_xlabel('Number of Projects')
        ax1.set_title('Project Categories', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10, fontweight='bold')

    # 2. Technology Usage (Top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    if technology_counts:
        techs = list(technology_counts.keys())
        tech_counts = list(technology_counts.values())
        sorted_tech_data = sorted(zip(techs, tech_counts), key=lambda x: x[1], reverse=True)
        sorted_techs, sorted_tech_counts = zip(*sorted_tech_data[:10])  # Top 10

        bars = ax2.barh(range(len(sorted_techs)), sorted_tech_counts, color='lightcoral')
        ax2.set_yticks(range(len(sorted_techs)))
        ax2.set_yticklabels(sorted_techs, fontsize=9)
        ax2.set_xlabel('Number of Projects')
        ax2.set_title('Technology Usage', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for i, (bar, count) in enumerate(zip(bars, sorted_tech_counts)):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', fontsize=10, fontweight='bold')

    # 3. Quality Distribution (Top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    quality_scores = [proj['quality_metrics']['overall_quality'] for proj in project_analysis.values()]
    if quality_scores:
        ax3.hist(quality_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Overall Quality Score')
        ax3.set_ylabel('Number of Projects')
        ax3.set_title('Project Quality Distribution', fontweight='bold')
        ax3.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(quality_scores):.2f}')
        ax3.legend()

    # 4. GitHub Repository Analysis (Middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    github_stats = {
        'Has GitHub': sum(1 for proj in project_analysis.values() if proj['quality_metrics']['has_github']),
        'No GitHub': sum(1 for proj in project_analysis.values() if not proj['quality_metrics']['has_github'])
    }
    if sum(github_stats.values()) > 0:
        ax4.pie(github_stats.values(), labels=github_stats.keys(), autopct='%1.1f%%', startangle=90)
        ax4.set_title('GitHub Repository Presence', fontweight='bold')

    # 5. Quality vs Documentation (Middle-center)
    ax5 = fig.add_subplot(gs[1, 1])
    if project_analysis:
        doc_scores = [proj['quality_metrics']['documentation_score'] for proj in project_analysis.values()]
        complete_scores = [proj['quality_metrics']['completeness_score'] for proj in project_analysis.values()]

        scatter = ax5.scatter(doc_scores, complete_scores, alpha=0.6, s=60)
        ax5.set_xlabel('Documentation Score')
        ax5.set_ylabel('Completeness Score')
        ax5.set_title('Documentation vs Completeness', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Add trend line
        if len(doc_scores) > 1:
            z = np.polyfit(doc_scores, complete_scores, 1)
            p = np.poly1d(z)
            ax5.plot(sorted(doc_scores), p(sorted(doc_scores)), "r--", alpha=0.8)

    # 6. Category Complexity (Middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    if project_analysis:
        # Calculate average description length by category
        category_complexity = defaultdict(list)
        for user_name, proj in project_analysis.items():
            for cat in proj['categories']:
                # Find the project text for this user
                for project in projects:
                    if project['user_name'] == user_name:
                        category_complexity[cat].append(len(project['project_text']))
                        break

        if category_complexity:
            categories = list(category_complexity.keys())[:8]  # Top 8 categories
            avg_lengths = [np.mean(category_complexity[cat]) for cat in categories]

            bars = ax6.bar(range(len(categories)), avg_lengths, color='lightblue')
            ax6.set_xticks(range(len(categories)))
            ax6.set_xticklabels([cat.replace(' ', '\n') for cat in categories], fontsize=8, rotation=45)
            ax6.set_ylabel('Avg Description Length')
            ax6.set_title('Category Complexity\n(by description length)', fontweight='bold')

    # 7. Technology Combinations (Bottom - spans full width)
    ax7 = fig.add_subplot(gs[2, :])
    if project_analysis:
        # Find common technology combinations
        tech_combos = defaultdict(int)
        for proj in project_analysis.values():
            techs = proj['technologies']
            if len(techs) >= 2:
                # Sort to ensure consistent combination names
                combo = ' + '.join(sorted(techs[:3]))  # Limit to 3 for readability
                if combo:
                    tech_combos[combo] += 1

        if tech_combos:
            # Show top 10 combinations
            sorted_combos = sorted(tech_combos.items(), key=lambda x: x[1], reverse=True)[:10]
            combo_names, combo_counts = zip(*sorted_combos)

            bars = ax7.barh(range(len(combo_names)), combo_counts, color='gold')
            ax7.set_yticks(range(len(combo_names)))
            ax7.set_yticklabels([name.replace(' + ', '\n+ ') for name in combo_names], fontsize=10)
            ax7.set_xlabel('Number of Projects')
            ax7.set_title('Popular Technology Combinations', fontweight='bold')
            ax7.grid(axis='x', alpha=0.3)

            for i, (bar, count) in enumerate(zip(bars, combo_counts)):
                ax7.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        str(count), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('sidequests/community-projects/outputs/project_analysis_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    print("Comprehensive visualization saved to: sidequests/community-projects/outputs/project_analysis_comprehensive.png")


def save_detailed_analysis(project_analysis: Dict, projects: List[Dict], 
                          category_counts: Dict, technology_counts: Dict, timeline_data: List[Dict]):
    """Save comprehensive project analysis to JSON file."""

    # Calculate summary statistics
    total_projects = len(project_analysis)
    projects_with_github = sum(1 for proj in project_analysis.values() if proj['quality_metrics']['has_github'])
    avg_quality = np.mean([proj['quality_metrics']['overall_quality'] for proj in project_analysis.values()])
    avg_categories = np.mean([len(proj['categories']) for proj in project_analysis.values()])
    avg_technologies = np.mean([len(proj['technologies']) for proj in project_analysis.values()])

    detailed_analysis = {
        'summary': {
            'total_projects': total_projects,
            'projects_with_github': projects_with_github,
            'github_percentage': (projects_with_github / total_projects * 100) if total_projects > 0 else 0,
            'average_quality_score': avg_quality,
            'average_categories_per_project': avg_categories,
            'average_technologies_per_project': avg_technologies,
            'unique_categories': len(category_counts),
            'unique_technologies': len(technology_counts),
            'analysis_date': datetime.now().isoformat()
        },
        'category_distribution': category_counts,
        'technology_distribution': technology_counts,
        'timeline_data': timeline_data,
        'projects': []
    }

    # Add detailed project information
    for project in projects:
        user_name = project['user_name']
        if user_name in project_analysis:
            analysis = project_analysis[user_name]

            detailed_analysis['projects'].append({
                'user_name': user_name,
                'user_id': project['user_id'],
                'project_text': project['project_text'],
                'github_urls': project.get('github_urls', []),
                'categories': analysis['categories'],
                'technologies': analysis['technologies'],
                'quality_metrics': analysis['quality_metrics'],
                'created_at': project['created_at'],
                'channel_name': project.get('channel_name', 'Unknown'),
                'reaction_counts': project.get('reaction_counts', {}),
                'attachments_count': len(project.get('attachments', []))
            })

    # Save to file
    output_file = 'sidequests/community-projects/outputs/detailed_project_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed analysis saved to: {output_file}")

    # Print summary to console
    print("\n" + "="*80)
    print("PROJECT ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Projects: {total_projects}")
    print(f"Projects with GitHub: {projects_with_github} ({projects_with_github/total_projects*100:.1f}%)")
    print(f"Average Quality Score: {avg_quality:.2f}")
    print(f"Average Categories per Project: {avg_categories:.1f}")
    print(f"Average Technologies per Project: {avg_technologies:.1f}")

    print("\Top Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cat}: {count}")

    print("\Top Technologies:")
    for tech, count in sorted(technology_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {tech}: {count}")


def main():
    """Main function to run the project analysis."""

    # Load projects
    projects = load_projects('sidequests/community-projects/outputs/all_projects_export.json')
    print(f"Loaded {len(projects)} projects from all_projects_export.json")

    # Analyze projects
    category_counts, technology_counts, project_analysis, timeline_data = analyze_projects(projects)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(category_counts, technology_counts, project_analysis, timeline_data, projects)

    # Save detailed analysis
    save_detailed_analysis(project_analysis, projects, category_counts, technology_counts, timeline_data)

    print("\nProject analysis complete! Check the generated visualizations and detailed analysis file.")


if __name__ == "__main__":
    main()
