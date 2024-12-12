from src.tools import get_financial_statements

# Get NVIDIA's quarterly statements
statements = get_financial_statements('NVDA', 'income', 'quarterly', 4)

print("\nNVIDIA Quarterly Revenue:")
print("-" * 40)

for statement in statements.get('income_statements', []):
    period = statement.get('report_period', 'N/A')
    revenue = statement.get('revenue', 0)
    revenue_billions = revenue / 1_000_000_000  # Convert to billions
    print(f"Period: {period} | Revenue: ${revenue_billions:.2f}B") 