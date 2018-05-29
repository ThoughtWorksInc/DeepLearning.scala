libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq(
      "com.thoughtworks.deeplearning.etl" %% "cifar100" % "0.2.0",
      "ch.qos.logback" % "logback-classic" % "1.2.3" % Optional,
      "org.nd4j" %% "nd4s" % "0.8.0",
      "org.nd4j" % "nd4j-api" % "0.8.0",
      "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Optional
    )
  }
}

fork in Test := true

enablePlugins(JmhPlugin)

publishArtifact := false

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-bangnotation" % "1.0.0-RC10")

addCompilerPlugin("com.thoughtworks.dsl" %% "compilerplugins-reseteverywhere" % "1.0.0-RC10")

libraryDependencies += "com.thoughtworks.dsl" %% "domains-scalaz" % "1.0.0-RC10"

addCompilerPlugin("com.thoughtworks.import" %% "import" % "2.0.2")
