import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from src.document_loader import load_and_create_vector_store
from src.rag_pipeline import RAGPipeline

class PublicationExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ReadyTensor Publication Explorer")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        self.current_article_id = None
        self.rag_pipeline = None

        # Init RAG pipeline
        try:
            self.rag_pipeline = RAGPipeline()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize query system: {e}. Please check API key.")
            return

        # Load data and create title-to-ID mapping
        try:
            self.vectorstore, self.titles, self.articles = load_and_create_vector_store()
            self.title_to_id = {pub["title"]: pub["id"] for pub in self.vectorstore._collection.get()["metadatas"]}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load publications: {e}. Please check the data directory.")
            self.titles = []
            self.articles = {}
            self.title_to_id = {}
            return

        self.main_screen()

    def main_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ttk.Label(frame, text="Publications", font=("Arial", 16, "bold")).grid(row=0, column=0, pady=10)

        canvas_frame = ttk.Frame(frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        def update_canvas(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas_width = canvas.winfo_width()
            if canvas_width > 0:
                canvas.itemconfig(canvas_window, width=canvas_width)

        scrollable_frame.bind("<Configure>", update_canvas)
        canvas.bind("<Configure>", update_canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        scrollable_frame.grid_columnconfigure(0, weight=1)

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        style = ttk.Style()
        style.configure("TitleButton.TButton", font=("Arial", 12), padding=10)
        style.map("TitleButton.TButton",
                  background=[("active", "#e0e0e0")],
                  relief=[("active", "raised")])

        if not self.titles:
            ttk.Label(frame, text="No publications available.", font=("Arial", 12)).grid(row=1, column=0, pady=20)
        for i, title in enumerate(self.titles):
            btn = ttk.Button(
                scrollable_frame,
                text=title,
                style="TitleButton.TButton",
                command=lambda t=title: self.publication_screen(t)
            )
            btn.grid(row=i, column=0, sticky="ew", pady=5, padx=10)

    def publication_screen(self, title):
        self.current_article_id = self.title_to_id.get(title)
        if not self.current_article_id or self.current_article_id not in self.articles:
            messagebox.showerror("Error", f"Selected article '{title}' not found.")
            return

        for widget in self.root.winfo_children():
            widget.destroy()

        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        ttk.Button(main_frame, text="Back", command=self.main_screen).pack(pady=5, anchor="nw")

        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=3)

        ttk.Label(left_frame, text=title, font=("Arial", 14, "bold"), wraplength=600).pack(pady=5)
        text_area = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Arial", 12), height=20)
        text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_area.insert(tk.END, self.articles.get(self.current_article_id, "No content available"))
        text_area.config(state="disabled")

        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)

        ttk.Label(right_frame, text="Ask about this publication", font=("Arial", 12, "bold")).pack(pady=5)

        self.chat_history = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Arial", 10), height=15, state="disabled")
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.input_box = ttk.Entry(right_frame, font=("Arial", 10))
        self.input_box.pack(fill=tk.X, padx=5, pady=5)
        self.input_box.bind("<Return>", self.submit_query)

        ttk.Button(right_frame, text="Ask", command=self.submit_query).pack(pady=5)

        # Initialize RAG chain
        try:
            if not self.articles.get(self.current_article_id):
                messagebox.showwarning("Warning", "This publication has no content for querying.")
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1, "filter": {"id": self.current_article_id}})
            retrieved_docs = retriever.get_relevant_documents("test query")
            if retrieved_docs and any(doc.metadata["id"] != self.current_article_id for doc in retrieved_docs):
                messagebox.showerror("Error", "Query system retrieved incorrect article data.")
                return
            self.rag_pipeline.create_chain(retriever)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize query system: {e}.")
            self.rag_pipeline.chain = None

    def submit_query(self, event=None):
        query = self.input_box.get().strip()
        if not query:
            return

        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, f"You: {query}\n")

        if self.rag_pipeline is None or self.rag_pipeline.chain is None:
            self.chat_history.insert(tk.END, "Agent: Sorry, the query system is not initialized. Please try another publication.\n\n")
        else:
            try:
                response = self.rag_pipeline.chain.invoke({"query": query})
                response_text = response if isinstance(response, str) else response.get("result", str(response))
                self.chat_history.insert(tk.END, f"Agent: {response_text}\n\n")
            except Exception as e:
                self.chat_history.insert(tk.END, f"Agent: Sorry, an error occurred while processing your query: {e}. Please try again.\n\n")
                print(f"Query error: {e}")

        self.chat_history.see(tk.END)
        self.chat_history.config(state="disabled")
        self.input_box.delete(0, tk.END)

def main():
    root = tk.Tk()
    app = PublicationExplorerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()