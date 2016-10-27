function get_weights(dataloader, opt)
    n_pos = 0
    n_neg = 0
    for n, sample in dataloader:run() do
        n_pos = n_pos + torch.eq(sample.target, 1):sum()
        n_neg = n_neg + torch.eq(sample.target, 0):sum()
        if n > 20 then break end -- we check the first 20 minibatches, not more
    end

    -- reversed! weight is inversely proportional to occurence
    n_neg_w = n_pos / (n_pos + n_neg)
    n_pos_w = n_neg / (n_pos + n_neg)

    return {n_pos_w, n_neg_w}
end

return get_weights
