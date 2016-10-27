function createThreadPool(nThreads)
    threads = require 'threads'
    pool = threads.Threads(
       nThreads,
       function(threadid)
          gmsg = msg -- get it the msg upvalue and store it in thread state
       end
    )
    return pool
end

return createThreadPool
