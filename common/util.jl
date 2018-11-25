using LinearAlgebra

function preprocess(text)
    text = lowercase(text)
    text = replace(text, "." => " .")
    words = split(text, " ")
    word_to_id = Dict{String, Int}()
    id_to_word = Dict{Int, String}()
    for word in words
        word = String(word)
        if !haskey(word_to_id, word)
            new_id = length(word_to_id) + 1
            word_to_id[word] = new_id
            id_to_word[new_id] = word
        end
    end
    corpus = [word_to_id[String(w)] for w in words]
    return corpus, word_to_id, id_to_word
end

function create_co_matrix(corpus, vocab_size; window_size=1)
    corpus_size = length(corpus)
    co_matrix = zeros(Int64, vocab_size, vocab_size)
    
    for (idx, word_id) in enumerate(corpus)
        for i in 1:window_size
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 1
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            end
                
            if right_idx <= corpus_size
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
            end
        end
    end
    
    return co_matrix
end

function cos_similarity(x, y; ϵ=1e-8)
    nx = x ./ sqrt(sum(x .^ 2) + ϵ)
    ny = y ./ sqrt(sum(y .^ 2) + ϵ)
    return dot(nx, ny)
end

function most_similiar(query, word_to_id, id_to_word, word_matrix; top=5)
    if !haskey(word_to_id, query)
        println("$query is not found")
        return
    end
    
    # 1. get query info
    println("\n[$query]")
    qid = word_to_id[query]
    qvec = word_matrix[qid, :]
    
    # 2. compute cosine similarity
    vocab_size = length(id_to_word)
    similarity = zeros(vocab_size)
    for i in 1:vocab_size
        similarity[i] = cos_similarity(word_matrix[i, :], qvec)
    end
    
    # 3. print up to 'top' words
    count = 0
    for i in sortperm(-1 .* similarity)
        if id_to_word[i] == query
            continue
        end
        println("  $(id_to_word[i]): $(similarity[i])")
        count += 1
        if count >= top
            return
        end
    end
end