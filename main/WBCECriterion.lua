local WBCECriterion, parent = torch.class('nn.WBCECriterion', 'nn.Criterion')

local eps = 1e-12

function WBCECriterion:__init(skew, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end

    assert(#skew == 2, "skew should be table with two elements (+ weight, - weight)")
    self.skew = skew
end


function WBCECriterion:updateOutput(input, target)
    -- - skew(1) * log(input) * target - skew(2) * log(1 - input) * (1 - target)

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local weights = self.weights
    local skew = self.skew
    local output

    buffer:resizeAs(input)

    -- skew[1] * log(input) * target
    buffer:add(input, eps):log():mul(skew[1])

    output = torch.dot(target, buffer)

    -- skew[2] * log(1 - input) * (1 - target)
    buffer:mul(input, -1):add(1):add(eps):log():mul(skew[2])

    output = output + torch.sum(buffer)
    output = output - torch.dot(target, buffer)

    if self.sizeAverage then
        output = output / input:nElement()
    end

    self.output = - output

    return self.output
end

function WBCECriterion:updateGradInput(input, target)
    -- - (target - input) / ( input (1 - input) )
    -- The gradient is slightly incorrect:
    -- It should have be divided by (input + eps) (1 - input + eps)
    -- but it is divided by input (1 - input + eps) + eps
    -- This modification requires less memory to be computed.

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local buffer2 = input.new()
    local gradInput = self.gradInput
    local skew = self.skew

    buffer:resizeAs(input)
    -- 1 - x + eps
    buffer:add(-input, 1):add(eps)

    gradInput:resizeAs(input)
    -- skew(2) * (1 - y) / (1 - x + eps)
    gradInput:add(-target, 1):mul(skew[2]):cdiv(buffer)

    -- - skew(1) * y / (x + eps)
    buffer2:resizeAs(input)
    buffer2:add(input, 0):add(eps)
    buffer:zero()
    buffer:add(-skew[1], target):cdiv(buffer2)

    -- - skew(1) * y / (x + eps) - skew(2) * (1 - y) / ((1 - x) + eps)
    gradInput:add(buffer)

    if self.sizeAverage then
        gradInput:div(target:nElement())
    end

    return gradInput
end
