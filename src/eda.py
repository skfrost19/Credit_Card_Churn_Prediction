import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_visualization_environment():
    """Setup matplotlib and seaborn for optimal visualization."""
    logger.info("Setting up visualization environment...")
    
    # Set figure parameters
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    logger.info("Visualization environment setup complete")

def create_output_directory(output_dir='visualisations'):
    """Create output directory for visualizations if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_dir}")
    return output_dir

def analyze_churn_distribution(df, output_dir='visualisations'):
    """
    Analyze and visualize the distribution of churn vs non-churn customers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Analyzing churn distribution...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    churn_counts = df['Churn'].value_counts()
    sns.countplot(data=df, x='Churn', ax=axes[0])
    axes[0].set_title('Distribution of Churn vs Non-Churn Customers')
    axes[0].set_xlabel('Churn Status (0=No, 1=Yes)')
    axes[0].set_ylabel('Number of Customers')
    
    # Add count labels on bars
    for i, count in enumerate(churn_counts.values):
        axes[0].text(i, count + 10, str(count), ha='center', va='bottom')
    
    # Pie chart
    churn_percentages = df['Churn'].value_counts(normalize=True) * 100
    labels = ['No Churn', 'Churn']
    colors = ['lightblue', 'lightcoral']
    
    axes[1].pie(churn_percentages.values, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[1].set_title('Churn Rate Distribution')
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'churn_distribution.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Churn distribution plot saved to {filename}")
    
    # Log statistics
    logger.info(f"Churn statistics:")
    logger.info(f"  Total customers: {len(df)}")
    logger.info(f"  Churned customers: {churn_counts[1]} ({churn_percentages[1]:.1f}%)")
    logger.info(f"  Non-churned customers: {churn_counts[0]} ({churn_percentages[0]:.1f}%)")
    
    plt.show()
    plt.close()

def analyze_age_gender_by_churn(df, output_dir='visualisations'):
    """
    Analyze age and gender patterns by churn status.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Analyzing age and gender patterns by churn status...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Age distribution by churn
    sns.histplot(data=df, x='Age', hue='Churn', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Churn Status')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    
    # Age boxplot by churn
    sns.boxplot(data=df, x='Churn', y='Age', ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Churn Status (Boxplot)')
    axes[0, 1].set_xlabel('Churn Status (0=No, 1=Yes)')
    axes[0, 1].set_ylabel('Age')
    
    # Gender distribution by churn
    gender_churn = pd.crosstab(df['Gender'], df['Churn'], normalize='index') * 100
    gender_churn.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Churn Rate by Gender')
    axes[1, 0].set_xlabel('Gender')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].legend(['No Churn', 'Churn'])
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Gender count by churn
    sns.countplot(data=df, x='Gender', hue='Churn', ax=axes[1, 1])
    axes[1, 1].set_title('Customer Count by Gender and Churn Status')
    axes[1, 1].set_xlabel('Gender')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'age_gender_churn_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Age and gender analysis plot saved to {filename}")
    
    # Log statistics
    logger.info("Age and gender statistics by churn:")
    for churn_val in [0, 1]:
        churn_label = "Non-churned" if churn_val == 0 else "Churned"
        subset = df[df['Churn'] == churn_val]
        logger.info(f"  {churn_label} customers:")
        logger.info(f"    Average age: {subset['Age'].mean():.1f}")
        logger.info(f"    Age range: {subset['Age'].min()}-{subset['Age'].max()}")
        logger.info(f"    Gender distribution: {subset['Gender'].value_counts().to_dict()}")
    
    plt.show()
    plt.close()

def analyze_balance_salary_distributions(df, output_dir='visualisations'):
    """
    Analyze balance and salary distributions by churn status.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Analyzing balance and salary distributions...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Balance distribution by churn
    sns.histplot(data=df, x='Balance', hue='Churn', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Balance Distribution by Churn Status')
    axes[0, 0].set_xlabel('Balance')
    axes[0, 0].set_ylabel('Count')
    
    # Balance boxplot by churn
    sns.boxplot(data=df, x='Churn', y='Balance', ax=axes[0, 1])
    axes[0, 1].set_title('Balance Distribution by Churn Status (Boxplot)')
    axes[0, 1].set_xlabel('Churn Status (0=No, 1=Yes)')
    axes[0, 1].set_ylabel('Balance')
    
    # Salary distribution by churn
    sns.histplot(data=df, x='EstimatedSalary', hue='Churn', kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Estimated Salary Distribution by Churn Status')
    axes[1, 0].set_xlabel('Estimated Salary')
    axes[1, 0].set_ylabel('Count')
    
    # Salary boxplot by churn
    sns.boxplot(data=df, x='Churn', y='EstimatedSalary', ax=axes[1, 1])
    axes[1, 1].set_title('Estimated Salary Distribution by Churn Status (Boxplot)')
    axes[1, 1].set_xlabel('Churn Status (0=No, 1=Yes)')
    axes[1, 1].set_ylabel('Estimated Salary')
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'balance_salary_distributions.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Balance and salary distributions plot saved to {filename}")
    
    # Log statistics
    logger.info("Balance and salary statistics by churn:")
    for churn_val in [0, 1]:
        churn_label = "Non-churned" if churn_val == 0 else "Churned"
        subset = df[df['Churn'] == churn_val]
        logger.info(f"  {churn_label} customers:")
        logger.info(f"    Average balance: ${subset['Balance'].mean():,.2f}")
        logger.info(f"    Median balance: ${subset['Balance'].median():,.2f}")
        logger.info(f"    Average salary: ${subset['EstimatedSalary'].mean():,.2f}")
        logger.info(f"    Median salary: ${subset['EstimatedSalary'].median():,.2f}")
    
    plt.show()
    plt.close()

def analyze_product_credit_active_patterns(df, output_dir='visualisations'):
    """
    Analyze number of products, credit card, and active member patterns by churn.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Analyzing product, credit card, and active member patterns...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Number of products distribution
    df['NumOfProducts'] = df['NumOfProducts'].astype(int)
    sns.countplot(data=df, x='NumOfProducts', hue='Churn', ax=axes[0, 0])
    axes[0, 0].set_title('Number of Products by Churn Status')
    axes[0, 0].set_xlabel('Number of Products')
    axes[0, 0].set_ylabel('Count')
    
    # Number of products churn rate
    products_churn = pd.crosstab(df['NumOfProducts'], df['Churn'], normalize='index') * 100
    products_churn[1].plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Churn Rate by Number of Products')
    axes[0, 1].set_xlabel('Number of Products')
    axes[0, 1].set_ylabel('Churn Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Credit card distribution
    sns.countplot(data=df, x='HasCrCard', hue='Churn', ax=axes[0, 2])
    axes[0, 2].set_title('Credit Card Ownership by Churn Status')
    axes[0, 2].set_xlabel('Has Credit Card (0=No, 1=Yes)')
    axes[0, 2].set_ylabel('Count')
    
    # Active member distribution
    sns.countplot(data=df, x='IsActiveMember', hue='Churn', ax=axes[1, 0])
    axes[1, 0].set_title('Active Membership by Churn Status')
    axes[1, 0].set_xlabel('Is Active Member (0=No, 1=Yes)')
    axes[1, 0].set_ylabel('Count')
    
    # Credit card churn rate
    cc_churn = pd.crosstab(df['HasCrCard'], df['Churn'], normalize='index') * 100
    cc_churn[1].plot(kind='bar', ax=axes[1, 1], color='lightblue')
    axes[1, 1].set_title('Churn Rate by Credit Card Ownership')
    axes[1, 1].set_xlabel('Has Credit Card (0=No, 1=Yes)')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    # Active member churn rate
    active_churn = pd.crosstab(df['IsActiveMember'], df['Churn'], normalize='index') * 100
    active_churn[1].plot(kind='bar', ax=axes[1, 2], color='lightgreen')
    axes[1, 2].set_title('Churn Rate by Active Membership')
    axes[1, 2].set_xlabel('Is Active Member (0=No, 1=Yes)')
    axes[1, 2].set_ylabel('Churn Rate (%)')
    axes[1, 2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'product_credit_active_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Product, credit card, and active member analysis plot saved to {filename}")
    
    # Log statistics
    logger.info("Product, credit card, and active member statistics:")
    logger.info(f"  Products distribution: {df['NumOfProducts'].value_counts().sort_index().to_dict()}")
    logger.info(f"  Credit card ownership: {df['HasCrCard'].value_counts().to_dict()}")
    logger.info(f"  Active membership: {df['IsActiveMember'].value_counts().to_dict()}")
    
    # Churn rates
    logger.info("Churn rates by category:")
    logger.info(f"  By number of products: {products_churn[1].to_dict()}")
    logger.info(f"  By credit card: {cc_churn[1].to_dict()}")
    logger.info(f"  By active membership: {active_churn[1].to_dict()}")
    
    plt.show()
    plt.close()

def analyze_tenure_patterns(df, output_dir='visualisations'):
    """
    Analyze customer tenure patterns by churn status.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Analyzing customer tenure patterns...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Tenure distribution by churn
    sns.histplot(data=df, x='Tenure', hue='Churn', kde=True, ax=axes[0])
    axes[0].set_title('Tenure Distribution by Churn Status')
    axes[0].set_xlabel('Tenure (Years)')
    axes[0].set_ylabel('Count')
    
    # Tenure boxplot by churn
    sns.boxplot(data=df, x='Churn', y='Tenure', ax=axes[1])
    axes[1].set_title('Tenure Distribution by Churn Status (Boxplot)')
    axes[1].set_xlabel('Churn Status (0=No, 1=Yes)')
    axes[1].set_ylabel('Tenure (Years)')
    
    # Tenure bins analysis
    df_temp = df.copy()
    df_temp['TenureBins'] = pd.cut(df_temp['Tenure'], 
                                   bins=[0, 2, 5, 8, 10], 
                                   labels=['0-2 years', '2-5 years', '5-8 years', '8-10 years'])
    
    tenure_churn = pd.crosstab(df_temp['TenureBins'], df_temp['Churn'], normalize='index') * 100
    tenure_churn[1].plot(kind='bar', ax=axes[2], color='orange')
    axes[2].set_title('Churn Rate by Tenure Groups')
    axes[2].set_xlabel('Tenure Groups')
    axes[2].set_ylabel('Churn Rate (%)')
    axes[2].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'tenure_patterns.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Tenure patterns plot saved to {filename}")
    
    # Log statistics
    logger.info("Tenure statistics by churn:")
    for churn_val in [0, 1]:
        churn_label = "Non-churned" if churn_val == 0 else "Churned"
        subset = df[df['Churn'] == churn_val]
        logger.info(f"  {churn_label} customers:")
        logger.info(f"    Average tenure: {subset['Tenure'].mean():.1f} years")
        logger.info(f"    Median tenure: {subset['Tenure'].median():.1f} years")
        logger.info(f"    Tenure range: {subset['Tenure'].min()}-{subset['Tenure'].max()} years")
    
    plt.show()
    plt.close()

def create_correlation_heatmap(df, output_dir='visualisations'):
    """
    Create feature correlation heatmap.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    """
    logger.info("Creating feature correlation heatmap...")
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()

    logger.info(f"Correlation matrix:\n{correlation_matrix}")
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    filename = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Correlation heatmap saved to {filename}")
    plt.show()
    plt.close()

def generate_comprehensive_eda_report(df, output_dir='visualisations'):
    """
    Generate a comprehensive EDA report with all visualizations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_dir (str): Directory to save visualizations
    
    Returns:
        dict: Summary statistics and insights
    """
    logger.info("="*60)
    logger.info("STARTING COMPREHENSIVE EDA ANALYSIS")
    logger.info("="*60)
    
    # Setup environment and create output directory
    setup_visualization_environment()
    create_output_directory(output_dir)
    
    # Basic dataset information
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    # Generate all visualizations
    logger.info("\n" + "-"*40)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("-"*40)
    
    try:
        # 1. Churn distribution analysis
        analyze_churn_distribution(df, output_dir)
        
        # 2. Age and gender analysis
        analyze_age_gender_by_churn(df, output_dir)
        
        # 3. Balance and salary distributions
        analyze_balance_salary_distributions(df, output_dir)
        
        # 4. Product, credit card, and active member analysis
        analyze_product_credit_active_patterns(df, output_dir)
        
        # 5. Tenure patterns
        analyze_tenure_patterns(df, output_dir)
        
        # 6. Correlation heatmap
        create_correlation_heatmap(df, output_dir)
        
        logger.info("\n" + "="*60)
        logger.info("EDA ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"All visualizations saved to: {output_dir}")
        
        # Generate summary insights
        insights = {
            'total_customers': len(df),
            'churn_rate': (df['Churn'].sum() / len(df)) * 100,
            'avg_age': df['Age'].mean(),
            'avg_balance': df['Balance'].mean(),
            'avg_salary': df['EstimatedSalary'].mean(),
            'avg_tenure': df['Tenure'].mean(),
            'visualizations_saved': [
                'churn_distribution.png',
                'age_gender_churn_analysis.png',
                'balance_salary_distributions.png',
                'product_credit_active_analysis.png',
                'tenure_patterns.png',
                'correlation_heatmap.png'
            ]
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error during EDA analysis: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("EDA module loaded successfully!")
    
    # Example usage:
    # df = pd.read_csv('data/cleaned_data.csv')
    # insights = generate_comprehensive_eda_report(df, 'visualisations')
