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
