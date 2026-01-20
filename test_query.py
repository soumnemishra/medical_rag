from src.pubmed_client import PubMedClient

c = PubMedClient()
query = '(("Hirschsprung Disease"[tiab]) AND (treatment[tiab] OR therapy[tiab] OR management[tiab] OR Therapeutics[MeSH])) AND (Humans[MeSH])'
print(f"Testing query: {query[:80]}...")
docs = c.search(query, max_results=5)
print(f"Found {len(docs)} documents")
for d in docs[:3]:
    print(f"  - {d.title[:80]}")
