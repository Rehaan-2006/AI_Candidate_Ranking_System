"""
Shared constants used by both features.py and reasoning.py.
"""

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "hexaware", "ltimindtree", "l&t infotech", "birlasoft",
}


def is_consulting(company_name):
    """Returns True if company_name matches a known IT-services/consulting firm."""
    return any(firm in company_name.lower() for firm in CONSULTING_FIRMS)