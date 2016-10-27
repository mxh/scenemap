require 'cudnn'

function create_loss(opt)
    local sm = cudnn.SpatialSoftMax():cuda()

    function multi_classification_loss(input, target)
        mm, labels_input = torch.max(input, 2)

        inp_tar_eq       = torch.eq(labels_input, target)
        inp_tar_ne       = torch.ne(labels_input, target)

        correct          = torch.sum(inp_tar_eq)
        correct_rate     = correct / labels_input:nElement()

        incorrect        = torch.sum(inp_tar_ne)
        incorrect_rate   = incorrect / labels_input:nElement()

        return {correct_rate, incorrect_rate}

    end

    function multi_classification_loss_log(stats)
        return ('Acc %1.4f'):format(stats[1]) 
    end

    local loss = {}
    loss.compute = multi_classification_loss
    loss.log = multi_classification_loss_log

    return loss
end

return create_loss
