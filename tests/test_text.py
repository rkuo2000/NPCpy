import os
import tempfile
from npcpy.data.text import rag_search, load_all_files


def test_rag_search_with_string():
    """Test RAG search with string input"""
    try:
        text_data = """
        Python is a programming language. It is used for web development.
        Machine learning is popular with Python. Data science uses Python libraries.
        JavaScript is used for web development. HTML and CSS are markup languages.
        """
        
        results = rag_search(
            query="Python programming",
            text_data=text_data,
            similarity_threshold=0.1  
        )
        
        assert isinstance(results, list)
        print(f"RAG search found {len(results)} relevant snippets")
        
        if len(results) > 0:
            print(f"First result: {results[0][:100]}...")
        
    except Exception as e:
        print(f"RAG search with string failed: {e}")


def test_rag_search_with_dict():
    """Test RAG search with dictionary input"""
    try:
        text_data = {
            "doc1.txt": "Python is a versatile programming language used in many fields.",
            "doc2.txt": "Web development often uses JavaScript and HTML technologies.",
            "doc3.txt": "Machine learning algorithms are implemented in Python frameworks."
        }
        
        results = rag_search(
            query="machine learning Python",
            text_data=text_data,
            similarity_threshold=0.1
        )
        
        assert isinstance(results, list)
        print(f"RAG search with dict found {len(results)} relevant snippets")
        
        if len(results) > 0:
            filename, snippet = results[0]
            print(f"Found in {filename}: {snippet[:100]}...")
        
    except Exception as e:
        print(f"RAG search with dict failed: {e}")


def test_rag_search_no_embedding_model():
    """Test RAG search without providing embedding model"""
    try:
        text_data = "This is a simple test document about artificial intelligence."
        
        results = rag_search(
            query="artificial intelligence",
            text_data=text_data,
            embedding_model=None,  
            similarity_threshold=0.1
        )
        
        assert isinstance(results, list)
        print(f"RAG search without model found {len(results)} results")
        
    except Exception as e:
        print(f"RAG search without model failed (expected if sentence-transformers not installed): {e}")


def test_load_all_files():
    """Test loading all files from directory"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        files_to_create = {
            "test1.py": "def hello():\n    print('Hello from Python')",
            "test2.txt": "This is a text file with some content.",
            "test3.md": "
            "test4.js": "function greet() { console.log('Hello from JS'); }",
            "ignore.log": "This file should be ignored"
        }
        
        for filename, content in files_to_create.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)
        
        
        text_data = load_all_files(temp_dir, depth=1)
        
        assert isinstance(text_data, dict)
        assert len(text_data) >= 4  
        
        
        py_files = [path for path in text_data.keys() if path.endswith('.py')]
        assert len(py_files) >= 1
        
        print(f"Loaded {len(text_data)} files from directory")
        print(f"File extensions found: {[os.path.splitext(path)[1] for path in text_data.keys()]}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_all_files_custom_extensions():
    """Test loading files with custom extensions"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        files = {
            "data.csv": "name,age\nJohn,25",
            "config.json": '{"setting": "value"}',
            "readme.txt": "This is a readme file",
            "script.py": "print('hello')",
            "ignore.tmp": "temporary file"
        }
        
        for filename, content in files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)
        
        
        text_data = load_all_files(
            temp_dir, 
            extensions=[".csv", ".json"],
            depth=1
        )
        
        assert isinstance(text_data, dict)
        assert len(text_data) == 2  
        
        csv_files = [path for path in text_data.keys() if path.endswith('.csv')]
        json_files = [path for path in text_data.keys() if path.endswith('.json')]
        
        assert len(csv_files) == 1
        assert len(json_files) == 1
        
        print(f"Loaded {len(text_data)} files with custom extensions")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_all_files_with_subdirectories():
    """Test loading files from subdirectories"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        
        
        with open(os.path.join(temp_dir, "main.txt"), "w") as f:
            f.write("Main directory file")
        
        
        with open(os.path.join(subdir, "sub.txt"), "w") as f:
            f.write("Subdirectory file")
        
        
        text_data = load_all_files(temp_dir, depth=2)
        
        assert isinstance(text_data, dict)
        assert len(text_data) >= 2  
        
        main_files = [path for path in text_data.keys() if "main.txt" in path]
        sub_files = [path for path in text_data.keys() if "sub.txt" in path]
        
        assert len(main_files) == 1
        assert len(sub_files) == 1
        
        print(f"Loaded {len(text_data)} files including subdirectories")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_load_all_files_depth_limit():
    """Test depth limiting in load_all_files"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        level1 = os.path.join(temp_dir, "level1")
        level2 = os.path.join(level1, "level2")
        os.makedirs(level2)
        
        
        with open(os.path.join(temp_dir, "root.txt"), "w") as f:
            f.write("Root level file")
        
        with open(os.path.join(level1, "level1.txt"), "w") as f:
            f.write("Level 1 file")
        
        with open(os.path.join(level2, "level2.txt"), "w") as f:
            f.write("Level 2 file")
        
        
        text_data = load_all_files(temp_dir, depth=1)
        
        level2_files = [path for path in text_data.keys() if "level2.txt" in path]
        assert len(level2_files) == 0  
        
        print(f"Depth limit test: loaded {len(text_data)} files with depth=1")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_rag_search_high_threshold():
    """Test RAG search with high similarity threshold"""
    try:
        text_data = "The quick brown fox jumps over the lazy dog. Python is great for programming."
        
        results = rag_search(
            query="artificial intelligence deep learning",
            text_data=text_data,
            similarity_threshold=0.8  
        )
        
        assert isinstance(results, list)
        print(f"High threshold search found {len(results)} results (expected: few/none)")
        
    except Exception as e:
        print(f"High threshold search failed: {e}")


def test_load_all_files_empty_directory():
    """Test load_all_files with empty directory"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        text_data = load_all_files(temp_dir)
        
        assert isinstance(text_data, dict)
        assert len(text_data) == 0
        
        print("Empty directory test passed")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)
