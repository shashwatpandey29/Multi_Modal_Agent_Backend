from brain.brain import ResearchBrain

brain = ResearchBrain()
brain.ingest("sample.pdf")

print(brain.summarize())
print(brain.teach())
