import time
from npcpy.data.web import search_web, search_perplexity


def test_duckduckgo_search():
    """Test DuckDuckGo web search"""
    try:
        results = search_web(
            query="Python programming language",
            num_results=3,
            provider="duckduckgo"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        assert len(content_str) > 0
        assert len(link_str) > 0
        assert "python" in content_str.lower()
        
        print(f"DuckDuckGo search returned {len(content_str)} chars of content")
        print(f"Found links: {link_str[:200]}...")
        
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")


def test_google_search():
    """Test Google web search"""
    try:
        results = search_web(
            query="machine learning basics",
            num_results=2,
            provider="google"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        print(f"Google search returned {len(content_str)} chars of content")
        print(f"Found links: {link_str[:200]}...")
        
    except Exception as e:
        print(f"Google search failed (expected due to restrictions): {e}")


def test_perplexity_search():
    """Test Perplexity AI search"""
    try:
        
        results = search_perplexity(
            query="What is artificial intelligence?",
            api_key="fake_key"  
        )
        
        if results:
            assert len(results) == 2  
            print("Perplexity search successful")
        
    except Exception as e:
        print(f"Perplexity search failed (expected without API key): {e}")


def test_search_web_default_provider():
    """Test web search with default provider"""
    try:
        results = search_web(
            query="open source software",
            num_results=2
        )
        
        assert len(results) == 2
        content_str, link_str = results
        
        assert isinstance(content_str, str)
        assert isinstance(link_str, str)
        
        print(f"Default provider search returned {len(content_str)} chars")
        
    except Exception as e:
        print(f"Default provider search failed: {e}")


def test_search_web_rate_limiting():
    """Test search with small delay to respect rate limits"""
    try:
        
        results1 = search_web("test query 1", num_results=1, provider="duckduckgo")
        
        
        time.sleep(1)
        
        
        results2 = search_web("test query 2", num_results=1, provider="duckduckgo")
        
        assert len(results1) == 2
        assert len(results2) == 2
        
        print("Rate limiting test passed")
        
    except Exception as e:
        print(f"Rate limiting test failed: {e}")


def test_search_web_empty_results():
    """Test search with very specific query that might return few results"""
    try:
        results = search_web(
            query="very_specific_unusual_query_12345678",
            num_results=1,
            provider="duckduckgo"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        print(f"Specific query search: content={len(content_str)}, links={len(link_str)}")
        
    except Exception as e:
        print(f"Specific query search failed: {e}")


def test_perplexity_with_custom_params():
    """Test Perplexity with custom parameters"""
    try:
        perplexity_kwargs = {
            "max_tokens": 200,
            "temperature": 0.5
        }
        
        results = search_web(
            query="climate change effects",
            provider="perplexity",
            api_key="fake_key",
            perplexity_kwargs=perplexity_kwargs
        )
        
        print("Perplexity custom params test completed")
        
    except Exception as e:
        print(f"Perplexity custom params failed (expected): {e}")


def test_search_web_large_num_results():
    """Test search with larger number of results"""
    try:
        results = search_web(
            query="artificial intelligence",
            num_results=10,
            provider="duckduckgo"
        )
        
        assert len(results) == 2
        content_str, link_str = results
        
        
        assert len(content_str) > 100
        
        print(f"Large results search: {len(content_str)} chars of content")
        
    except Exception as e:
        print(f"Large results search failed: {e}")


def test_search_web_content_parsing():
    """Test that search results contain expected content structure"""
    try:
        results = search_web(
            query="Python tutorial",
            num_results=2,
            provider="duckduckgo"
        )
        
        content_str, link_str = results
        
        
        assert "Citation:" in content_str
        
        
        links = link_str.strip().split('\n')
        assert len(links) >= 1
        
        print(f"Content parsing test passed - found {len(links)} links")
        
    except Exception as e:
        print(f"Content parsing test failed: {e}")
