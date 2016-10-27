function create_loss(opt)
    function nop_loss(input, target)
        return {}
    end

    function nop_loss_log(stats)
        return ''
    end

    local loss = {}
    loss.compute = nop_loss
    loss.log = nop_loss_log

    return loss
end

return create_loss
