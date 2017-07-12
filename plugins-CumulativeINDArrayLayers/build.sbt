enablePlugins(Example)

libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2, 12)) {
    Nil
  } else {
    Seq("org.nd4j" % "nd4j-native-platform" % "0.8.0" % Test)
  }
}

fork in Test := true

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1" % Test
