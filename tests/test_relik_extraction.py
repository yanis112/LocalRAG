from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
print(RelikOutput)