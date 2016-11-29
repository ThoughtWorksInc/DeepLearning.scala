addSbtPlugin("org.scala-js" % "sbt-scalajs" % "0.6.13")

addSbtPlugin("com.thoughtworks.sbt-best-practice" % "sbt-best-practice" % "latest.release")

addSbtPlugin("com.thoughtworks.deeplearning" % "sbt-nd4j" % "1.0.0-M2")

resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.5")
