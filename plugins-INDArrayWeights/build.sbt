libraryDependencies += "com.thoughtworks.feature" %% "partialapply" % "2.1.0-M0"

libraryDependencies += "com.thoughtworks.feature" %% "implicitapply" % "2.1.0-M0"

libraryDependencies += "com.thoughtworks.feature" %% "factory" % "2.1.0-M0"

libraryDependencies += "com.thoughtworks.feature" %% "mixins-implicitssingleton" % "2.1.0-M0"

libraryDependencies += "com.chuusai" %% "shapeless" % "2.3.2"

libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" %% "nd4s" % "0.8.0",
        "org.nd4j" % "nd4j-api" % "0.8.0",
        "org.nd4j" % "nd4j-native-platform" % "0.8.0" % Test)
  }
}

fork in Test := true

enablePlugins(Example)