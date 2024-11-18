import subprocess
from src.main_utils.generation_utils_v2 import LLM_answer_v3

class GraphMaker:
    def __init__(self, model_name='gpt-4o',llm_provider='github'):
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.exemple_code_1 = """vars: {
  d2-config: {
    layout-engine: elk
    theme-id: 300
  }
}
pipeline: {
  user: {
    shape: person
    width: 130
  }

  "Query Preprocessing": {
    chunking: {
      shape: rectangle
    }
  }

  "Lexical Search": {
    shape: cylinder
  }

  "Semantic Search": {
    shape: cylinder
  }

  reranker: {
    shape: diamond
  }

  retriever: {
    shape: parallelogram
  }

  "Generation Model": {
    shape: parallelogram
    style.fill: lightgreen
  }

  "Knowledge Base": {
    shape: stored_data
    style.multiple: true
  }

  user -> "Query Preprocessing".chunking: "submit query"
  "Query Preprocessing".chunking -> "Lexical Search": "process tokens"
  "Query Preprocessing".chunking -> "Semantic Search": "process embeddings"

  "Lexical Search" -> reranker: "lexical results"
  "Semantic Search" -> reranker: "semantic results"

  reranker -> retriever: "ranked results"
  retriever -> "Knowledge Base": "fetch documents"
  "Knowledge Base" -> retriever: "documents"

  retriever -> "Generation Model": "retrieve context"
  "Generation Model" -> user: "generate response"
}"""
        self.exemple_code_2 = """
        Cluster: {
  grid-rows: 2
  vertical-gap: 10

  label: "Temporal Cluster"
  style.fill: transparent
  style.stroke-dash: 3
  style.double-border: false

  Server: {
    grid-columns: 3
    horizontal-gap: 120

    style.fill: transparent
    style.stroke-dash: 3

    History: {
      style.multiple: true
    }
    Matching: {
      style.multiple: true
    }
    Frontend: {
      style.multiple: true
    }
  }
  Database: {
    label: ""
    style.opacity: 0

    grid-columns: 3
    horizontal-gap: 120
    vertical-gap: 10

    placeho1.style.opacity: 0 # hack to align Database
    Database: {
      label: "Database"
      style.border-radius: 100
    }
  }
}

Outside: {
  grid-rows: 2
  vertical-gap: 10
  horizontal-gap: 20
  style.opacity: 0

  Worker: {
    style.multiple: true
  }
  placeho.style.opacity: 0 # hack to align Worker
}

Outside.Worker -> Cluster.Server.Frontend: Poll Tasks
Cluster.Server.Frontend -> Cluster.Server.Matching: Poll Tasks
Cluster.Server.History -> Cluster.Server.Matching: Add Tasks
Cluster.Server.Matching <-> Cluster.Database.Database"""


        self.exemple_code_3="""
        user01: {
  label: User01
  class: [base; person; multiple]
}

user02: {
  label: User02
  class: [base; person; multiple]
}

user03: {
  label: User03
  class: [base; person; multiple]
}

user01 -> container.task01: {
  label: Create Task
  class: [base; animated]
}
user02 -> container.task02: {
  label: Create Task
  class: [base; animated]
}
user03 -> container.task03: {
  label: Create Task
  class: [base; animated]
}

container: Application {
  direction: right
  style: {
    bold: true
    font-size: 28
  }
  icon: https://icons.terrastruct.com/dev%2Fgo.svg

  task01: {
    icon: https://icons.terrastruct.com/essentials%2F092-graph%20bar.svg
    class: [task; multiple]
  }

  task02: {
    icon: https://icons.terrastruct.com/essentials%2F095-download.svg
    class: [task; multiple]
  }

  task03: {
    icon: https://icons.terrastruct.com/essentials%2F195-attachment.svg
    class: [task; multiple]
  }

  queue: {
    label: Queue Library
    icon: https://icons.terrastruct.com/dev%2Fgo.svg
    style: {
      bold: true
      font-size: 32
      fill: honeydew
    }

    producer: {
      label: Producer
      class: library
    }

    consumer: {
      label: Consumer
      class: library
    }

    database: {
      label: Ring\nBuffer
      shape: cylinder
      style: {
        bold: true
        font-size: 32
        fill-pattern: lines
        font: mono
      }
    }

    producer -> database
    database -> consumer
  }

  worker01: {
    icon: https://icons.terrastruct.com/essentials%2F092-graph%20bar.svg
    class: [task]
  }

  worker02: {
    icon: https://icons.terrastruct.com/essentials%2F095-download.svg
    class: [task]
  }

  worker03: {
    icon: https://icons.terrastruct.com/essentials%2F092-graph%20bar.svg
    class: [task]
  }

  worker04: {
    icon: https://icons.terrastruct.com/essentials%2F195-attachment.svg
    class: [task]
  }

  task01 -> queue.producer: {
    class: [base; enqueue]
  }
  task02 -> queue.producer: {
    class: [base; enqueue]
  }
  task03 -> queue.producer: {
    class: [base; enqueue]
  }
  queue.consumer -> worker01: {
    class: [base; dispatch]
  }
  queue.consumer -> worker02: {
    class: [base; dispatch]
  }
  queue.consumer -> worker03: {
    class: [base; dispatch]
  }
  queue.consumer -> worker04: {
    class: [base; dispatch]
  }
}
"""

    def generate_graph(self, base_prompt, output_svg='output.svg'):
        generation_prompt=f""" Based on the following instructions, your goal is to generate a code in .d2 format that represents a graph. The code should be generated in a way that it can be used by the D2 software to generate a visual representation of the graph. \
            The instructions are as follows: {base_prompt}. Here are some an exemples of a working .d2 codes: EX1: {self.exemple_code_1}, EX2: {self.exemple_code_2}, EX3: {self.exemple_code_3}. Drawing inspiration from all those exemples codes and only using components, colours, elements , ect, explicitely used in them, you will answer with the code in .d2 format without any preamble."""
            
        # Appeler le LLM pour obtenir le code .d2
        d2_code = LLM_answer_v3(
            prompt=generation_prompt,
            model_name=self.model_name,
            llm_provider=self.llm_provider,
        )
        
        #process the code by replaxing: ``` by nothing
        d2_code=d2_code.replace("```","")
        
        print("##############################################")
        print("GENERATED D2 CODE:")
        print(d2_code)
        print("##############################################")
        # Sauvegarder le code dans input.d2
        with open('input.d2', 'w', encoding='utf-8') as file:
            file.write(d2_code)
        # Générer l'image SVG
        subprocess.run([
            r"C:\Program Files\D2\d2.exe",
            "--sketch",
            "input.d2",
            output_svg,
        ], shell=True)
        
    def convert_svg_to_png(self,input_svg='output.svg', output_png='output.png'):
      import cairosvg
      cairosvg.svg2png(url=input_svg, write_to=output_png)
      
    def show_graph(self, output_svg='output.svg'):
        import os
        import webbrowser
        # Open the SVG file directly in the default browser
        webbrowser.open(f'file://{os.path.abspath(output_svg)}')
  
    # def show_graph(self, output_svg='output.svg'):
    #     import os
    #     import matplotlib
    #     matplotlib.use('TkAgg')  # Set interactive backend
    #     import matplotlib.pyplot as plt
    #     import matplotlib.image as mpimg
        
    #     # Convert SVG to PNG
    #     output_png = output_svg.replace('.svg', '.png')
    #     self.convert_svg_to_png(input_svg=output_svg, output_png=output_png)
        
    #     # Read and display PNG
    #     img = mpimg.imread(output_png)
    #     plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.imshow(img)
    #     plt.show(block=True)  # Make sure window stays open
            
if __name__ == '__main__':
    graph_maker = GraphMaker(model_name='Meta-Llama-3.1-405B-Instruct',llm_provider='github')
    prompt = "A graph for a cooking recipe chatbot application. The graph should include the following components: User, Recipe Database, Recipe Retrieval, Recipe Generation, Recipe Display. The User should be connected to the Recipe Database, the Recipe Retrieval, and the Recipe Generation. The Recipe Database should be connected to the Recipe Retrieval. The Recipe Retrieval should be connected to the Recipe Generation. The Recipe Generation should be connected to the Recipe Display. The Recipe Database should be a stored data component. The Recipe Generation should have a lightgreen fill color."
    graph_maker.generate_graph(prompt)
    graph_maker.show_graph()