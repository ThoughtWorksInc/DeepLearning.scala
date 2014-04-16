Immutable Future
================

`immutable-future` is a set of DSL for asynchronous programming, in the pure functional favor.

## Usage

    import scala.concurrent.duration._
    import scala.util.control.Exception.Catcher
    import com.qifun.immutableFuture.Future
    
    val executor = java.util.concurrent.Executors.newSingleThreadScheduledExecutor
    
    // Manually implements an immutable future, which is the asynchronous version of `Thread.sleep()`
    def asyncSleep(duration: Duration) = new Future[Unit] {
      import scala.util.control.TailCalls._
      def onComplete(handler: Unit => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]) = {
        executor.schedule(new Runnable {
          def run() {
            handler().result
          }
        }, duration.length, duration.unit)
        done()
      }
    }
    
    // Without the keyword `new`, you have the magic version of `Future` constructor,
    // which enables the magic postfix `await`.
    val sleep10seconds = Future {
      var i = 0
      while (i < 10) {
        println(s"I have sleeped $i times")
        // The magic postfix `await` invokes asynchronous method like normal `Thread.sleep()`,
        // and does not block any thread.
        asyncSleep(1.seconds).await
        i += 1
      }
      i
    }
    
    // When `sleep10seconds` is running, it could report failture to this catcher
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    
    // An immutable future instance is lazy, only evaluating when you query it.
    println("Before evaluation of the immutable future `sleep10seconds`")
    for (total <- sleep10seconds) {
      println("After evaluation of the immutable future `sleep10seconds`")
      println(s"I sleeped $total times in total.")
      executor.shutdown()
    }


Run it and you will see the output:

    Before evaluation of the immutable future `sleep10seconds`
    I have sleeped 0 times
    I have sleeped 1 times
    I have sleeped 2 times
    I have sleeped 3 times
    I have sleeped 4 times
    I have sleeped 5 times
    I have sleeped 6 times
    I have sleeped 7 times
    I have sleeped 8 times
    I have sleeped 9 times
    After evaluation of the immutable future `sleep10seconds`
    I sleeped 10 times in total.

### Explaination

There are two sorts of API to use a immutable future, the for-comprehensions style API and "A-Normal Form" style API.

#### For-Comprehensions

The for-comprehensions style API for `immutable-future` is like the [for-comprehensions for scala.concurrent.Future](http://docs.scala-lang.org/overviews/core/futures.html#functional_composition_and_forcomprehensions). 

    for (total <- sleep10seconds) {
      println("After evaluation of the immutable future `sleep10seconds`")
      println(s"I sleeped $total times in total.")
      executor.shutdown()
    }

A notable difference between the for-comprehensions for immutable futures and for `scala.concurrent.Future`s is the required implicit parameter. `scala.concurrent.Future` requires an `ExecutionContext`, while immutable future requires a `Catcher`.

    import scala.util.control.Exception.Catcher
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }

